# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MenloConfig:
    menlo_api_key: str
    tokamak_api_key: str
    rcs_url: str = "https://api.menlo.ai/rcs"
    viewer_base_url: str = "https://sim.menlo.ai"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def load_config(require_tokamak: bool = False) -> MenloConfig:
    """Load configuration from .env or the current environment."""
    _load_dotenv()

    config = MenloConfig(
        menlo_api_key=os.environ.get("MENLO_API_KEY", ""),
        tokamak_api_key=os.environ.get("TOKAMAK_API_KEY", ""),
        rcs_url=os.environ.get("MENLO_RCS_URL", "https://api.menlo.ai/rcs"),
        viewer_base_url=os.environ.get("MENLO_VIEWER_BASE_URL", "https://sim.menlo.ai"),
    )

    if not config.menlo_api_key or config.menlo_api_key.startswith("sk_live_your"):
        raise RuntimeError(
            "MENLO_API_KEY is not set. Add it to .env or export it in your shell."
        )
    if require_tokamak and not config.tokamak_api_key:
        raise RuntimeError(
            "TOKAMAK_API_KEY is required for this program. Add it to .env or export it."
        )
    return config


