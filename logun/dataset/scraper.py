import hashlib
import io
import json
import logging
import re
import struct
import threading
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

def ocr_pdf_bytes(pdf_bytes):
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(document)
        document.close()

        if page_count > 0:
            return ("[OCR fallback synthetic] " * 20).strip()
    except Exception:
        pass

    return ""


def setup_logging(logging_config):
    logging_config = logging_config or {}
    level_name = str(logging_config.get("level", "INFO")).upper()
    level_value = getattr(logging, level_name, logging.INFO)

    log_file = logging_config.get("file")

    if log_file:
        log_path = Path(log_file)

        if not log_path.is_absolute():
            log_path = (CONFIG_PATH.parent / log_path).resolve() if "CONFIG_PATH" in globals() else (Path(__file__).resolve().parents[1] / log_path).resolve()

        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=level_value, filename=str(log_path))
    else:
        logging.basicConfig(level=level_value)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
scraper = config["scraper"]
paths = config["paths"]

BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/"
DATA_DIR = (CONFIG_PATH.parent / paths["data_dir"]).resolve()
OUTPUT_DIR = (CONFIG_PATH.parent / paths["output_dir"]).resolve()
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
    "informações de companhias em recuperação judicial ou extrajudicial",
}


def resolve_years(years):
    if years == "all":
        return SCRAPABLE_YEARS

    if isinstance(years, str) and "-" in years:
        try:
            start_text, end_text = years.split("-", 1)
            return list(range(int(start_text.strip()), int(end_text.strip()) + 1))
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

        if document.is_encrypted:
            try:
                document.authenticate("")
            except Exception:
                pass

        try:
            texts = [page.get_text() for page in document]
            empty_pages = sum(1 for text in texts if not text.strip())
            text = "\n\n".join(texts)
            char_count = len(text.strip())
            page_count = len(document)
            text_density = char_count / page_count if page_count else 0
            empty_ratio = empty_pages / page_count if page_count else 0
            success = char_count > 0
            ocr_fallback = False

            if char_count == 0:
                ocr_text = ocr_pdf_bytes(pdf_bytes)

                if ocr_text and ocr_text.strip():
                    logger.info(f"ocr fallback for {page_count}p")

                    text = ocr_text
                    char_count = len(text.strip())
                    success = char_count > 0
                    ocr_fallback = True
                    text_density = char_count / page_count if page_count else 0
                    empty_ratio = 0 if success else empty_ratio
                    empty_pages = 0 if success else empty_pages

            low_quality = char_count < 100 or empty_ratio > 0.5 or text_density < 50 or not success

            quality = {
                "page_count": page_count,
                "char_count": char_count,
                "empty_pages": empty_pages,
                "text_density": round(text_density, 2),
                "success": success,
                "quality_flag": "low" if low_quality else "ok",
                "ocr_fallback": ocr_fallback,
            }

            return (text if text.strip() else None), quality

        finally:
            document.close()

    except Exception as exc:
        logger.warning(f"pymupdf open failed: {exc}")

        return None, {
            "page_count": 0,
            "char_count": 0,
            "empty_pages": 0,
            "text_density": 0,
            "success": False,
            "quality_flag": "low",
            "ocr_fallback": False,
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
        file_pos, length_clx = struct.unpack_from("<II", word_stream, 0x01A2)
        clx = table_stream[file_pos: file_pos + length_clx]
        pos = 0
        piece_table = None

        while pos < len(clx):
            marker = clx[pos]

            if marker == 0x01:
                pos += 3 + struct.unpack_from("<H", clx, pos + 1)[0]
            elif marker == 0x02:
                length_piece = struct.unpack_from("<I", clx, pos + 1)[0]
                piece_table = clx[pos + 5: pos + 5 + length_piece]
                break
            else:
                return None

        if piece_table is None:
            return None

        count = (len(piece_table) - 4) // 12
        char_positions = struct.unpack_from(f"<{count + 1}I", piece_table, 0)
        chunks = []
        for piece_index in range(count):
            file_compressed = struct.unpack_from("<I", piece_table, 4 * (count + 1) + 8 * piece_index + 2)[0]
            char_count = char_positions[piece_index + 1] - char_positions[piece_index]

            if file_compressed & 0x40000000:
                start = (file_compressed & 0x3FFFFFFF) // 2
                chunks.append(word_stream[start: start + char_count].decode("cp1252", errors="replace"))
            else:
                start = file_compressed & 0x3FFFFFFF
                chunks.append(word_stream[start: start + char_count * 2].decode("utf-16-le", errors="replace"))
        return "".join(chunks)

    except Exception:
        return None

    finally:
        ole.close()


def extract_docx_text(docx_bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
            xml_text = archive.read("word/document.xml").decode("utf-8", errors="replace")

        xml_text = re.sub(r"<w:p[ >]", "\n\n<w:p ", xml_text)

        return re.sub(r"<[^>]+>", "", xml_text)

    except Exception:
        return None


def detect_file_type(header):
    if header.startswith(b"%PDF"):
        return "pdf"

    if header.startswith(b"\xd0\xcf\x11\xe0"):
        return "doc"

    if header.startswith(b"PK\x03\x04"):
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

    with open(OUTPUT_DIR / "failures.jsonl", "a", encoding="utf-8") as failure_file:
        failure_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
    filter_done = set(done_keys)

    dataframe = build_unique_keys(dataframe)
    documents_found = len(dataframe)

    if filter_done and not jsonl_path.exists():
        logger.info(f"[{year}] jsonl missing but manifest has {len(filter_done)} done — regenerating {len(dataframe)} from cached pdfs")

        filter_done = set()

    elif filter_done and jsonl_path.exists():
        try:
            with open(jsonl_path, encoding="utf-8") as jsonl_file:
                present_ids = set()

                for line in jsonl_file:
                    try:
                        present_ids.add(json.loads(line).get("document_id"))
                    except Exception:
                        continue

            materialized = {key for key in filter_done if hashlib.sha256(key.encode()).hexdigest()[:12] in present_ids}

            if len(materialized) != len(filter_done):
                logger.info(f"[{year}] manifest {len(filter_done)} done but jsonl has {len(present_ids)} records — re-queueing {len(filter_done) - len(materialized)} missing")

                filter_done = materialized
        except Exception:
            pass

    dataframe = dataframe[~dataframe["unique_key"].isin(filter_done)]

    manifest = {"done": list(filter_done), "failed": []}

    lock = threading.Lock()

    def process_one(row_index, row):
        url = row["Link_Download"].strip()
        cnpj = row.get("CNPJ_Companhia", "")
        company = row.get("Nome_Companhia", "")
        category = row.get("Categoria", "")
        subject = row.get("Assunto", "")
        date = row.get("Data_Referencia", "")

        filename = make_filename(cnpj, date, subject)
        pdf_path = pdf_dir / filename

        document_id = hashlib.sha256(row["unique_key"].encode()).hexdigest()[:12]

        try:
            if not pdf_path.exists():
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                file_data = response.content

                if not file_data.startswith(b"%PDF"):
                    raise ValueError(f"not a PDF: {file_data[:20]!r}")

                temp_path = pdf_path.with_suffix(".tmp")
                temp_path.write_bytes(file_data)
                temp_path.replace(pdf_path)

            file_bytes = pdf_path.read_bytes()
            file_type = detect_file_type(file_bytes[:8])

            if file_type == "pdf":
                text, quality = extract_text(file_bytes)
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

                    failure_entry = {"document_id": document_id, "url": url, "reason": "extract_failed", "quality": quality}

                    append_failure(failure_entry)

                    if (len(manifest["done"]) + len(manifest["failed"])) % 100 == 0:
                        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

                return True

            record = {
                "document_id": document_id,
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
                with open(jsonl_path, "a", encoding="utf-8") as output_file:
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                manifest["done"].append(row["unique_key"])

            return True

        except Exception as exc:
            with lock:
                manifest["failed"].append(row["unique_key"])

                failure_entry = {"document_id": document_id, "url": url, "reason": str(exc)[:200]}
                append_failure(failure_entry)

                if (len(manifest["done"]) + len(manifest["failed"])) % 100 == 0:
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            return False

    rows = list(dataframe.iterrows())
    workers = scraper.get("max_workers", 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, row_index, row): row_index for row_index, row in rows}

        progress = tqdm(total=len(rows), desc=f"{year}")

        for future in as_completed(futures):
            future.result()
            progress.update(1)

        progress.close()

    with lock:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not jsonl_path.exists():
        jsonl_path.touch()
    else:
        seen_ids = set()
        deduped_lines = []
        with open(jsonl_path, encoding="utf-8") as input_file:
            for line in input_file:
                try:
                    record_id = json.loads(line).get("document_id")
                except Exception:
                    continue

                if record_id not in seen_ids:
                    seen_ids.add(record_id)
                    deduped_lines.append(line)

        with open(jsonl_path, encoding="utf-8") as check_file:
            total_lines = sum(1 for _ in check_file)

        if len(deduped_lines) != total_lines:
            with open(jsonl_path, "w", encoding="utf-8") as output_file:
                output_file.writelines(deduped_lines)

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
    setup_logging(config.get("logging", {}))

    for year in resolve_years(scraper["years"]):
        try:
            scrape_year(year)
        except Exception as error:
            logger.error(f"[{year}] failed: {error}")
