"""Load the YAML config and secrets from the environment."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline_config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_eia_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY is not set. Add it to your .env file.")
    return key
