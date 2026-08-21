import hashlib
import io
import json
import logging
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import pymupdf
import requests
import yaml
from tqdm import tqdm
logger = logging.getLogger(__name__)
def setup_logging(cfg):
    logging_cfg = cfg.get("logging", {})
    level_name = logging_cfg.get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    log_file = logging_cfg.get("file")
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = Path(__file__).parent / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=level, filename=str(log_path))
    else:
        logging.basicConfig(level=level)
CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
scraper = config["scraper"]
paths = config["paths"]
BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/"
DATA_DIR = Path(__file__).parent / paths["data_dir"]
AUDIT_DIR = Path(__file__).parent / "audit"
SCRAPABLE_YEARS = list(range(2003, 2027))
KEEP_CATEGORIES = {"fato relevante", "comunicado ao mercado", "aviso aos acionistas", "carta anual de governança corporativa", "código de conduta", "estatuto social", "política de remuneração", "política de divulgação de ato ou fato relevante", "política de gerenciamento de riscos", "política de negociação de valores mobiliários", "política de transações entre partes relacionadas", "política de destinação de resultados", "política de dividendos", "política de negociação das ações da companhia", "política de sustentabilidade", "política para contratação de serviços extra-auditoria de seus auditores independentes", "política sobre contribuições e doações", "plano de remuneração baseado em ações", "plano de remuneração baseado em ações (exceto plano de opções)", "regimento interno da diretoria", "regimento interno de comitês", "regimento interno do comitê de auditoria estatutário", "regimento interno do conselho fiscal", "regimento interno do conselho de administração", "relato integrado", "relatório de sustentabilidade", "dados econômico-financeiros", "contratos de identidade", "acordo de acionistas", "comunicação sobre transação entre partes relacionadas", "informações de companhias em recuperação judicial ou extrajudicial"}
rate_lock = threading.Lock()
last_request_ts = [0.0]
def global_rate_limit():
    delay = scraper.get("delay", 0) or 0
    if delay <= 0:
        return
    with rate_lock:
        now = time.monotonic()
        wait = last_request_ts[0] + delay - now
        if wait > 0:
            time.sleep(wait)
        last_request_ts[0] = time.monotonic()
def resolve_years(years):
    return SCRAPABLE_YEARS if years == "all" else years
def download_csv(year):
    year_dir = DATA_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    csv_path = year_dir / "ipe.csv"
    if csv_path.exists():
        return csv_path
    response = requests.get(f"{BASE_URL}ipe_cia_aberta_{year}.zip", timeout=60)
    response.raise_for_status()
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    csv_name = max((n for n in zip_file.namelist() if n.endswith(".csv")), key=lambda n: zip_file.getinfo(n).file_size)
    csv_path.write_bytes(zip_file.read(csv_name))
    return csv_path
def load_csv(csv_path):
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            dataframe = pd.read_csv(csv_path, dtype=str, encoding=encoding, sep=";", keep_default_na=False)
            if len(dataframe.columns) < 2:
                continue
            return dataframe
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise RuntimeError(f"Cannot parse {csv_path}")
def extract_text(pdf_bytes):
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            texts = [page.get_text() for page in doc]
            empty_pages = sum(1 for t in texts if not t.strip())
            text = "\n\n".join(texts)
            char_count = len(text.strip())
            page_count = len(doc)
            text_density = char_count / page_count if page_count else 0
            empty_ratio = empty_pages / page_count if page_count else 0
            success = char_count > 0
            low = char_count < 100 or empty_ratio > 0.5 or text_density < 50 or not success
            quality = {"page_count": page_count, "char_count": char_count, "empty_pages": empty_pages, "text_density": round(text_density, 2), "success": success, "quality_flag": "low" if low else "ok"}
            return (text if text.strip() else None), quality
        finally:
            doc.close()
    except Exception:
        return None, {"page_count": 0, "char_count": 0, "empty_pages": 0, "text_density": 0, "success": False, "quality_flag": "low"}
def build_unique_keys(dataframe):
    dataframe["unique_key"] = dataframe["Link_Download"].fillna("").str.strip() + "|" + dataframe["CNPJ_Companhia"].fillna("").str.strip() + "|" + dataframe["Data_Referencia"].fillna("").str.strip() + "|" + dataframe["Assunto"].fillna("").str.strip()
    return dataframe[dataframe["Link_Download"].fillna("").str.strip() != ""]
def make_filename(cnpj, date, subject):
    raw = f"{cnpj}_{date}_{subject[:50].replace(' ', '_')}.pdf"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)
def scrape_year(year):
    csv_path = download_csv(year)
    dataframe = load_csv(csv_path)
    category_column = next((c for c in dataframe.columns if "categ" in c.lower()), None)
    if category_column:
        dataframe[category_column] = dataframe[category_column].fillna("").str.strip().str.lower()
        dataframe = dataframe[dataframe[category_column].isin(KEEP_CATEGORIES)]
    if scraper["dry_run"]:
        logger.info(f"[{year}] {len(dataframe)} rows")
        return
    year_dir = DATA_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = year_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    jsonl_path = year_dir / f"{year}.jsonl"
    manifest_path = year_dir / "manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    done_keys = set(existing.get("done", []))
    dataframe = build_unique_keys(dataframe)
    documents_found = len(dataframe)
    dataframe = dataframe[~dataframe["unique_key"].isin(done_keys)]
    if scraper["max_docs"] > 0:
        dataframe = dataframe.sample(frac=1).head(scraper["max_docs"])
    manifest = {"done": list(done_keys), "failed": []}
    failures = []
    lock = threading.Lock()
    def process_one(index, row):
        url = row["Link_Download"].strip()
        cnpj = row.get("CNPJ_Companhia", "")
        company = row.get("Nome_Companhia", "")
        category = row.get("Categoria", "")
        subject = row.get("Assunto", "")
        date = row.get("Data_Referencia", "")
        filename = make_filename(cnpj, date, subject)
        pdf_path = pdf_dir / filename
        doc_id = hashlib.sha256(row["unique_key"].encode()).hexdigest()[:12]
        try:
            if not pdf_path.exists():
                global_rate_limit()
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                pdf_path.write_bytes(response.content)
            text, quality = extract_text(pdf_path.read_bytes())
            if not quality["success"]:
                with lock:
                    manifest["done"].append(row["unique_key"])
                    failures.append({"document_id": doc_id, "url": url, "reason": "extract_failed", "quality": quality})
                return True
            record = {"document_id": doc_id, "text": text, "company": company, "cnpj": cnpj, "category": category, "subject": subject, "date": date, "year": year, "source_url": url}
            with lock:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest["done"].append(row["unique_key"])
            return True
        except Exception as exc:
            with lock:
                manifest["failed"].append(row["unique_key"])
                failures.append({"document_id": doc_id, "url": url, "reason": str(exc)[:200]})
            return False
    rows = list(dataframe.iterrows())
    workers = scraper.get("max_workers", 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, i, row): i for i, row in rows}
        pbar = tqdm(total=len(rows), desc=f"{year}")
        for future in as_completed(futures):
            future.result()
            pbar.update(1)
        pbar.close()
    with lock:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    run_stats = {"year": year, "documents_found": documents_found, "documents_downloaded": len(manifest["done"]), "documents_failed": len(manifest["failed"])}
    (AUDIT_DIR / "run.json").write_text(json.dumps(run_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        with open(AUDIT_DIR / "failures.jsonl", "a", encoding="utf-8") as fh:
            for item in failures:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"[{year}] done: {len(manifest['done'])} downloaded, {len(manifest['failed'])} failed")
if __name__ == "__main__":
    setup_logging(config)
    for year in resolve_years(scraper["years"]):
        try:
            scrape_year(year)
        except Exception as error:
            logger.error(f"[{year}] failed: {error}")
