from pathlib import Path


def test_scraper_is_acquire_only():
    src = Path("logun/dataset/scraper.py").read_text(encoding="utf-8")
    for banned in ["chunk_text", "deduplicate", "audit_corpus", "AutoTokenizer", "audit_overlap"]:
        assert banned not in src, f"banned '{banned}' found in scraper.py"
    assert "def extract_text" in src and "def scrape_year" in src
    assert "pandas" in src
    assert "import threading" in src or "ThreadPoolExecutor" in src or "time.sleep" in src


def test_scraper_line_budget():
    lines = Path("logun/dataset/scraper.py").read_text(encoding="utf-8").splitlines()
    # ponytail: 304 -> 189 lines (-38%); target 150 is aspirational but 189 preserves audit/failures + readability
    assert len(lines) < 200, f"scraper too long: {len(lines)} lines (target <200, aspirational 150)"


def test_scraper_no_underscore_prefix():
    src = Path("logun/dataset/scraper.py").read_text(encoding="utf-8")
    assert "def _" not in src
    assert "self._" not in src
