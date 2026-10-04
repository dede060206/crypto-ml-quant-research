from pathlib import Path
import yaml


def settings():
    return yaml.safe_load((Path(__file__).resolve().parents[1]/'configs'/'training.yaml').read_text())
