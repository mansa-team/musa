import yaml
from pathlib import Path


def test_config_has_audit_and_logging():
    cfg = yaml.safe_load(Path("logun/dataset/config.yaml").read_text(encoding="utf-8"))
    assert "audit" in cfg and "logging" in cfg or "scraper" in cfg
    assert cfg["dapt"]["tokenizer"] == "answerdotai/ModernBERT-base"
    assert cfg["dapt"]["chunk_size"] == 4096


def test_logging_present():
    cfg = yaml.safe_load(Path("logun/dataset/config.yaml").read_text(encoding="utf-8"))
    assert "logging" in cfg
    assert cfg["logging"]["level"] == "INFO"
    assert cfg["logging"]["file"] == "logs/logun.log"


def test_config_mapping_comments():
    text = Path("logun/dataset/config.yaml").read_text(encoding="utf-8")
    assert "scraper -> cvm" in text
    assert "dapt -> processing" in text
    assert "paths -> output" in text
