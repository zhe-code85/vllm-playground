"""
Deployment Profile Store

Reads deployment profiles from the control-layer directory (`.env.*` files) so
that model / engine parameters come from a file instead of the web form.

A profile is the *same* `.env.qwen` file consumed by the control-layer shell
scripts (`_common.sh` sources it, `docker-compose.qwen.yml` interpolates it).
Keeping one file for both means the parameters cannot drift apart.

Keys are split three ways:

  HOST_KEYS       -> turned into `docker run` flags (image, name, mounts, ...)
  CONTAINER_KEYS  -> passed through verbatim as `-e KEY=VALUE`; the in-container
                     launcher `launch-qwen.sh` assembles the vllm argv from them
  IGNORED_KEYS    -> only meaningful to the control-layer shell scripts

Follows the same shape as SettingsStore / MCPConfigStore: module-level logger,
defensive parsing, never raises on a malformed file.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Directory scanned for `.env.*` profiles. Override with the environment
# variable when running Playground somewhere other than the deployment host.
DEFAULT_PROFILE_DIR = "/home/user/ai/control"

# Filename prefix that marks a profile, and suffixes that never are one.
PROFILE_PREFIX = ".env."
_NOT_A_PROFILE_SUFFIX = (".example", ".bak", ".sample", ".template")

# Keys consumed on the host side; each maps to a `docker run` flag rather than
# being injected into the container. See container_manager.build_container_config.
HOST_KEYS = {
    "QWEN_IMAGE",
    "CONTAINER_NAME",
    "MODEL_DIR",
    "VLLM_CACHE_DIR",
    "VLLM_PORT",
    "VLLM_BIND_HOST",
    "RESTART_POLICY",
    "STOP_GRACE_PERIOD",
    "LOG_MAX_SIZE",
    "LOG_MAX_FILE",
    "HEALTHCHECK_START_PERIOD",
    "SHM_SIZE",
}

# Keys handed to the container as-is. NVIDIA_VISIBLE_DEVICES appears here *and*
# is read on the host to build `--gpus`; it is legitimately needed in both places.
CONTAINER_KEYS = {
    "MODEL_PATH",
    "SERVED_MODEL_NAME",
    "TENSOR_PARALLEL_SIZE",
    "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION",
    "KV_CACHE_DTYPE",
    "KV_CACHE_MEMORY",
    "MM_ENCODER_ATTN_BACKEND",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "ENABLE_MTP",
    "MTP_NUM_TOKENS",
    "ENABLE_PREFIX_CACHING",
    "DISABLE_CUSTOM_ALL_REDUCE",
    "REASONING_PARSER",
    "ENABLE_AUTO_TOOL_CHOICE",
    "TOOL_CALL_PARSER",
    "VLLM_EXTRA_ARGS",
    "VLLM_API_KEY",
    "VLLM_LOGGING_LEVEL",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "HF_HUB_OFFLINE",
}

# Keys that only the control-layer shell scripts care about.
IGNORED_KEYS = {
    "PROJECT_NAME",
    "VLLM_HOST",
    "WAIT_ATTEMPTS",
    "WAIT_SECONDS",
    "TAIL",
    "SMOKE_CONCURRENCY",
}

# Values never sent to the browser.
SECRET_KEYS = {"VLLM_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MODELSCOPE_TOKEN"}

# docker-compose.qwen.yml hardcodes `MODEL_PATH: /model`, so the .env files do
# not carry it. Supply the same value here.
CONTAINER_DEFAULTS = {"MODEL_PATH": "/model"}

# Fallbacks matching the `${VAR:-default}` expressions in docker-compose.qwen.yml.
HOST_DEFAULTS = {
    "CONTAINER_NAME": "qwen-svc",
    "VLLM_PORT": "8000",
    "VLLM_BIND_HOST": "0.0.0.0",
    "RESTART_POLICY": "unless-stopped",
    "STOP_GRACE_PERIOD": "120s",
    "HEALTHCHECK_START_PERIOD": "900s",
    "LOG_MAX_SIZE": "64m",
    "LOG_MAX_FILE": "5",
}


def get_profile_dir() -> Path:
    """Directory profiles are read from."""
    return Path(os.environ.get("VLLM_PLAYGROUND_PROFILE_DIR", DEFAULT_PROFILE_DIR))


def _strip_quotes(value: str) -> str:
    """Remove one layer of matching quotes, the way `source` would."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_env_file(path: Path) -> Dict[str, str]:
    """
    Parse a shell-style `KEY=VALUE` file.

    Deliberately conservative and aligned with how `_common.sh` sources these
    files: comments and blank lines are skipped, a leading `export ` is
    tolerated, one layer of matching quotes is stripped, and no variable
    expansion is performed (the control-layer .env files contain no references).

    Empty values are dropped entirely rather than kept as "": `launch-qwen.sh`
    tests them with `[ -n "$X" ]`, so an empty value and an absent key mean the
    same thing, and dropping them keeps the container's environment honest.
    """
    parsed: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read profile {path}: {e}")
        return parsed

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            logger.warning(f"{path.name}:{lineno} is not KEY=VALUE, skipping: {raw!r}")
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = _strip_quotes(value.strip())
        if value == "":
            continue  # empty == absent, see docstring
        parsed[key] = value

    return parsed


def list_profiles() -> List[str]:
    """
    Profile names available in the profile directory.

    `.env.qwen` -> "qwen". Templates and backups are excluded.
    """
    profile_dir = get_profile_dir()
    if not profile_dir.is_dir():
        logger.info(f"Profile directory does not exist: {profile_dir}")
        return []

    names = []
    for entry in profile_dir.iterdir():
        if not entry.is_file() or not entry.name.startswith(PROFILE_PREFIX):
            continue
        name = entry.name[len(PROFILE_PREFIX) :]
        if not name or name.endswith(_NOT_A_PROFILE_SUFFIX):
            continue
        names.append(name)

    return sorted(names)


def load_profile(name: str) -> Optional[Dict[str, Any]]:
    """
    Load one profile, split into host-side and container-side settings.

    Returns None when the name is invalid or the file is missing, so callers can
    turn that into a 404 rather than a 500.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        logger.warning(f"Rejected profile name: {name!r}")
        return None

    path = get_profile_dir() / f"{PROFILE_PREFIX}{name}"
    if not path.is_file():
        logger.warning(f"Profile not found: {path}")
        return None

    parsed = _parse_env_file(path)

    host = dict(HOST_DEFAULTS)
    container = dict(CONTAINER_DEFAULTS)
    unknown = []

    for key, value in parsed.items():
        if key in HOST_KEYS:
            host[key] = value
        elif key in CONTAINER_KEYS:
            container[key] = value
        elif key in IGNORED_KEYS:
            continue
        else:
            # An unrecognised key is far more likely to be a newly added engine
            # flag than a typo, so pass it to the container instead of dropping
            # it silently -- but say so, since a typo would land here too.
            unknown.append(key)
            container[key] = value

    if unknown:
        logger.warning(
            f"Profile '{name}' has keys not in the known schema, passing to container: {', '.join(unknown)}"
        )

    return {"name": name, "path": str(path), "host": host, "container": container}


def redacted(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of *profile* safe to send to the browser."""

    def scrub(section: Dict[str, str]) -> Dict[str, str]:
        return {k: ("***" if k in SECRET_KEYS else v) for k, v in section.items()}

    return {
        "name": profile.get("name"),
        "path": profile.get("path"),
        "host": scrub(profile.get("host", {})),
        "container": scrub(profile.get("container", {})),
    }
