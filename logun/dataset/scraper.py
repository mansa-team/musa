import hashlib
import io
import json
import logging
import re
import struct
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pymupdf
import olefile
import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(config):
    logging_config = config.get("logging", {})
    level = getattr(logging, str(logging_config.get("level", "INFO")).upper(), logging.INFO)
    log_file = logging_config.get("file")

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
OUTPUT_DIR = Path(__file__).parent / paths["output_dir"]
AUDIT_DIR = OUTPUT_DIR
SCRAPABLE_YEARS = list(range(2003, 2027))
KEEP_CATEGORIES = {
    "fato relevante", "comunicado ao mercado", "aviso aos acionistas", "carta anual de governança corporativa", 
    "código de conduta", "estatuto social", "política de remuneração", "política de divulgação de ato ou fato relevante", 
    "política de gerenciamento de riscos", "política de negociação de valores mobiliários", "política de transações entre partes relacionadas", 
    "política de destinação de resultados", "política de dividendos", "política de negociação das ações da companhia", "política de sustentabilidade", 
    "política para contratação de serviços extra-auditoria de seus auditores independentes", "política sobre contribuições e doações", 
    "plano de remuneração baseado em ações", "plano de remuneração baseado em ações (exceto plano de opções)", "regimento interno da diretoria", 
    "regimento interno de comitês", "regimento interno do comitê de auditoria estatutário", "regimento interno do conselho fiscal", 
    "regimento interno do conselho de administração", "relato integrado", "relatório de sustentabilidade", "dados econômico-financeiros", 
    "contratos de identidade", "acordo de acionistas", "comunicação sobre transação entre partes relacionadas", 
    "informações de companhias em recuperação judicial ou extrajudicial"
    }
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
    if years == "all":
        return SCRAPABLE_YEARS

    if isinstance(years, str) and "-" in years:
        try:
            year_start, year_end = years.split("-", 1)

            return list(range(int(year_start.strip()), int(year_end.strip()) + 1))
        except Exception:
            return SCRAPABLE_YEARS

    return years


def download_csv(year):
    data_dir_path = DATA_DIR / str(year)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir_path / "ipe.csv"

    if csv_path.exists():
        return csv_path

    response = requests.get(f"{BASE_URL}ipe_cia_aberta_{year}.zip", timeout=60)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    csv_name = max(
        (name for name in zip_file.namelist() if name.endswith(".csv")),
        key=lambda name: zip_file.getinfo(name).file_size,
    )
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
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        try:
            texts = [page.get_text() for page in document]
            empty_pages = sum(1 for text in texts if not text.strip())
            text = "\n\n".join(texts)
            char_count = len(text.strip())
            page_count = len(document)
            text_density = char_count / page_count if page_count else 0
            empty_ratio = empty_pages / page_count if page_count else 0
            success = char_count > 0
            low = char_count < 100 or empty_ratio > 0.5 or text_density < 50 or not success
            quality = {
                "page_count": page_count,
                "char_count": char_count,
                "empty_pages": empty_pages,
                "text_density": round(text_density, 2),
                "success": success,
                "quality_flag": "low" if low else "ok",
            }

            return (text if text.strip() else None), quality
        finally:
            document.close()
    except Exception:
        return None, {
            "page_count": 0,
            "char_count": 0,
            "empty_pages": 0,
            "text_density": 0,
            "success": False,
            "quality_flag": "low",
        }


def extract_doc_text(doc_bytes):
    try:
        ole = olefile.OleFileIO(io.BytesIO(doc_bytes))
    except Exception:
        return None

    try:
        word_stream = ole.openstream("WordDocument").read()

        if struct.unpack_from("<H", word_stream, 0)[0] != 0xA5EC:
            return None

        flags = struct.unpack_from("<H", word_stream, 0x0A)[0]
        table_name = "1Table" if flags & 0x0200 else "0Table"

        if not ole.exists(table_name):
            table_name = "0Table" if ole.exists("0Table") else "1Table"

        table_stream = ole.openstream(table_name).read()
        fc_clx, lcb_clx = struct.unpack_from("<II", word_stream, 0x01A2)
        clx = table_stream[fc_clx : fc_clx + lcb_clx]

        pos = 0
        piece_table = None

        while pos < len(clx):
            clxt = clx[pos]

            if clxt == 0x01:
                pos += 3 + struct.unpack_from("<H", clx, pos + 1)[0]
            elif clxt == 0x02:
                lcb = struct.unpack_from("<I", clx, pos + 1)[0]
                piece_table = clx[pos + 5 : pos + 5 + lcb]
                break
            else:
                return None

        if piece_table is None:
            return None

        count = (len(piece_table) - 4) // 12
        cps = struct.unpack_from(f"<{count + 1}I", piece_table, 0)
        chunks = []

        for index in range(count):
            fc_compressed = struct.unpack_from("<I", piece_table, 4 * (count + 1) + 8 * index + 2)[0]
            char_count = cps[index + 1] - cps[index]

            if fc_compressed & 0x40000000:
                start = (fc_compressed & 0x3FFFFFFF) // 2
                chunks.append(word_stream[start : start + char_count].decode("cp1252", errors="replace"))
            else:
                start = fc_compressed & 0x3FFFFFFF
                chunks.append(word_stream[start : start + char_count * 2].decode("utf-16-le", errors="replace"))

        return "".join(chunks)
    except Exception:
        return None
    finally:
        ole.close()


def extract_docx_text(docx_bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="replace")

        xml = re.sub(r"<w:p[ >]", "\n\n<w:p ", xml)

        return re.sub(r"<[^>]+>", "", xml)
    except Exception:
        return None


def detect_file_type(head):
    if head.startswith(b"%PDF"):
        return "pdf"

    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "doc"

    if head.startswith(b"PK\x03\x04"):
        return "docx"

    return "unknown"


def document_quality(text):
    char_count = len(text or "")

    return {
        "page_count": 1,
        "char_count": char_count,
        "empty_pages": 0,
        "text_density": char_count,
        "success": char_count >= 100,
        "quality_flag": "ok" if char_count >= 100 else "low",
    }


def append_failure(entry):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "failures.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _pdf_worker_loop():
    while True:
        line = sys.stdin.readline()

        if not line:
            break

        path = line.strip()

        try:
            text, quality = extract_text(Path(path).read_bytes())
        except Exception:
            text = None
            quality = {"page_count": 0, "char_count": 0, "empty_pages": 0, "text_density": 0, "success": False, "quality_flag": "low"}

        sys.stdout.write(json.dumps([text, quality], ensure_ascii=False) + "\n")
        sys.stdout.flush()


class PdfExtractWorker:
    # ponytail: MuPDF segfaults natively on corrupt PDFs — isolate in a subprocess so the pipeline survives

    def __init__(self):
        self.process = None

    def _spawn(self):
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__)), "--pdf-worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def extract(self, path):
        if self.process is None or self.process.poll() is not None:
            self._spawn()

        result_line = ""

        try:
            self.process.stdin.write(str(path).replace("\n", " ") + "\n")
            self.process.stdin.flush()
            result_line = self.process.stdout.readline()
        except Exception:
            pass

        if not result_line:
            self.process.kill()
            self.process.wait()
            self._spawn()

            return None, {
                "page_count": 0,
                "char_count": 0,
                "empty_pages": 0,
                "text_density": 0,
                "success": False,
                "quality_flag": "low",
            }

        return json.loads(result_line)


pdf_local = threading.local()


def pdf_worker_for_thread():
    worker = getattr(pdf_local, "worker", None)

    if worker is None:
        worker = PdfExtractWorker()
        pdf_local.worker = worker

    return worker


def build_unique_keys(dataframe):
    dataframe["unique_key"] = (
        dataframe["Link_Download"].fillna("").str.strip()
        + "|"
        + dataframe["CNPJ_Companhia"].fillna("").str.strip()
        + "|"
        + dataframe["Data_Referencia"].fillna("").str.strip()
        + "|"
        + dataframe["Assunto"].fillna("").str.strip()
    )

    return dataframe[dataframe["Link_Download"].fillna("").str.strip() != ""]


def make_filename(cnpj, date, subject):
    raw = f"{cnpj}_{date}_{subject[:50].replace(' ', '_')}.pdf"

    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)


def scrape_year(year):
    csv_path = download_csv(year)
    dataframe = load_csv(csv_path)

    category_column = next((col for col in dataframe.columns if "categ" in col.lower()), None)

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

            file_bytes = pdf_path.read_bytes()
            file_type = detect_file_type(file_bytes[:8])

            if file_type == "pdf":
                text, quality = pdf_worker_for_thread().extract(str(pdf_path))
            elif file_type == "doc":
                text = extract_doc_text(file_bytes)
                quality = document_quality(text)
            elif file_type == "docx":
                text = extract_docx_text(file_bytes)
                quality = document_quality(text)
            else:
                text = None
                quality = {"success": False, "quality_flag": "unknown_format"}

            if not quality["success"]:
                with lock:
                    manifest["done"].append(row["unique_key"])
                    failures.append(
                        {"document_id": doc_id, "url": url, "reason": "extract_failed", "quality": quality}
                    )
                    append_failure(failures[-1])

                    if (len(manifest["done"]) + len(manifest["failed"])) % 100 == 0:
                        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

                return True

            record = {
                "document_id": doc_id,
                "text": text,
                "company": company,
                "cnpj": cnpj,
                "category": category,
                "subject": subject,
                "date": date,
                "year": year,
                "source_url": url,
                "extraction_quality": quality,
            }

            with lock:
                with open(jsonl_path, "a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

                manifest["done"].append(row["unique_key"])

            return True
        except Exception as exc:
            with lock:
                manifest["failed"].append(row["unique_key"])
                failures.append({"document_id": doc_id, "url": url, "reason": str(exc)[:200]})
                append_failure(failures[-1])

                if (len(manifest["done"]) + len(manifest["failed"])) % 100 == 0:
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            return False

    rows = list(dataframe.iterrows())
    workers = scraper.get("max_workers", 8)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, row_index, row): row_index for row_index, row in rows}
        pbar = tqdm(total=len(rows), desc=f"{year}")

        for future in as_completed(futures):
            future.result()
            pbar.update(1)

        pbar.close()

    with lock:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ponytail: crash between jsonl-write and manifest-append can duplicate records — dedupe by document_id, keep first
    seen_document_ids = set()
    deduped_lines = []

    with open(jsonl_path, encoding="utf-8") as file:
        for line in file:
            try:
                record_id = json.loads(line).get("document_id")
            except Exception:
                continue

            if record_id not in seen_document_ids:
                seen_document_ids.add(record_id)
                deduped_lines.append(line)

    if len(deduped_lines) != sum(1 for _ in open(jsonl_path, encoding="utf-8")):
        with open(jsonl_path, "w", encoding="utf-8") as file:
            file.writelines(deduped_lines)

        logger.info(f"[{year}] deduped shard: {len(deduped_lines)} unique records")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_stats = {
        "year": year,
        "documents_found": documents_found,
        "documents_downloaded": len(manifest["done"]),
        "documents_failed": len(manifest["failed"]),
    }
    (OUTPUT_DIR / "scrape_audit.json").write_text(json.dumps(run_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"[{year}] done: {len(manifest['done'])} downloaded, {len(manifest['failed'])} failed")


if __name__ == "__main__":
    setup_logging(config)

    if "--pdf-worker" in sys.argv:
        _pdf_worker_loop()
        sys.exit(0)

    for year in resolve_years(scraper["years"]):
        try:
            scrape_year(year)
        except Exception as error:
            logger.error(f"[{year}] failed: {error}")
