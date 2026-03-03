import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    af2_tar: Path
    features_targets: Path
    features_fingerprints: Path
    features_binders: Path
    structures_pdb: Path
    binders_human: Path
    binders_pdb: Path
    binders_viral: Path
    binders_merged: Path


def load_config(config_path: Path) -> Config:
    """Loads and validates a Mimir run config JSON file.

    Resolves all values inside the 'paths' sub-dict relative to the project
    root (the directory containing the config file) and converts them to Path objects.

    Args:
        config_path: Path to the config.json file.

    Returns:
        Config object with paths resolved to Path objects.

    Raises:
        SystemExit: If the file is missing, not valid JSON, or lacks a 'paths' key.
    """
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        sys.exit(1)

    if "paths" not in data:
        logger.error("Config file must contain a 'paths' key")
        sys.exit(1)

    paths_data = data["paths"]
    project_root = Path.cwd()

    required_keys = [
        "af2_tar",
        "features_targets",
        "features_fingerprints",
        "features_binders",
        "structures_pdb",
        "binders_human",
        "binders_pdb",
        "binders_viral",
        "binders_merged",
    ]

    for key in required_keys:
        if key not in paths_data:
            logger.error(f"Missing required path key in config: {key}")
            sys.exit(1)

    def resolve_path(value: str) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return project_root / p

    return Config(
        af2_tar=resolve_path(paths_data["af2_tar"]),
        features_targets=resolve_path(paths_data["features_targets"]),
        features_fingerprints=resolve_path(paths_data["features_fingerprints"]),
        features_binders=resolve_path(paths_data["features_binders"]),
        structures_pdb=resolve_path(paths_data["structures_pdb"]),
        binders_human=resolve_path(paths_data["binders_human"]),
        binders_pdb=resolve_path(paths_data["binders_pdb"]),
        binders_viral=resolve_path(paths_data["binders_viral"]),
        binders_merged=resolve_path(paths_data["binders_merged"]),
    )

