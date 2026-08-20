import json, time, zipfile, io
from pathlib import Path
import yaml
import pandas as pd
import pymupdf, requests
from tqdm import tqdm

CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
scraper = config["scraper"]
paths = config["paths"]

BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/"
DATA_DIR = Path(__file__).parent / paths["data_dir"]
SCRAPABLE_YEARS = list(range(2003, 2027))
KEEP_CATEGORIES = {
    "fato relevante", "comunicado ao mercado", "aviso aos acionistas",
    "carta anual de governança corporativa", "código de conduta", "estatuto social",
    "política de remuneração", "política de divulgação de ato ou fato relevante",
    "política de gerenciamento de riscos", "política de negociação de valores mobiliários",
    "política de transações entre partes relacionadas", "política de destinação de resultados",
    "política de dividendos", "política de negociação das ações da companhia",
    "política de sustentabilidade", "política para contratação de serviços extra-auditoria de seus auditores independentes",
    "política sobre contribuições e doações", "plano de remuneração baseado em ações",
    "plano de remuneração baseado em ações (exceto plano de opções)",
    "regimento interno da diretoria", "regimento interno de comitês",
    "regimento interno do comitê de auditoria estatutário", "regimento interno do conselho fiscal",
    "regimento interno do conselho de administração", "relato integrado",
    "relatório de sustentabilidade", "dados econômico-financeiros",
    "contratos de identidade", "acordo de acionistas",
    "comunicação sobre transação entre partes relacionadas",
    "informações de companhias em recuperação judicial ou extrajudicial",
}


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
        text = "\n\n".join(page.get_text() for page in document)
        document.close()
        return text if text.strip() else None
    except Exception:
        return None


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

    category_column = next(
        (col for col in dataframe.columns if "categ" in col.lower()), None
    )
    if category_column:
        dataframe[category_column] = dataframe[category_column].fillna("").str.strip().str.lower()
        dataframe = dataframe[dataframe[category_column].isin(KEEP_CATEGORIES)]

    if scraper["dry_run"]:
        print(f"[{year}] {len(dataframe)} rows")
        return

    year_dir = DATA_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = year_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    jsonl_path = year_dir / f"{year}.jsonl"
    manifest_path = year_dir / "manifest.json"
    done_keys = set(json.load(open(manifest_path))["done"]) if manifest_path.exists() else set()

    dataframe = build_unique_keys(dataframe)
    dataframe = dataframe[~dataframe["unique_key"].isin(done_keys)]

    if scraper["max_docs"] > 0:
        dataframe = dataframe.sample(frac=1).head(scraper["max_docs"])

    manifest = {"done": list(done_keys), "failed": []}

    with open(jsonl_path, "a", encoding="utf-8") as output_file:
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"{year}"):
            url = row["Link_Download"].strip()
            cnpj = row.get("CNPJ_Companhia", "")
            company = row.get("Nome_Companhia", "")
            category = row.get("Categoria", "")
            subject = row.get("Assunto", "")
            date = row.get("Data_Referencia", "")

            filename = make_filename(cnpj, date, subject)
            pdf_path = pdf_dir / filename

            try:
                if not pdf_path.exists():
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    pdf_path.write_bytes(response.content)
                    time.sleep(scraper["delay"])

                text = extract_text(pdf_path.read_bytes())
                if not text or len(text.strip()) < 100:
                    manifest["done"].append(row["unique_key"])
                    continue

                record = {
                    "cnpj": cnpj, "company": company, "category": category,
                    "subject": subject, "date": date, "year": year,
                    "source_url": url, "filename": filename, "text": text,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest["done"].append(row["unique_key"])

            except Exception:
                manifest["failed"].append(row["unique_key"])

            if (index + 1) % 50 == 0:
                json.dump(manifest, open(manifest_path, "w"))

    json.dump(manifest, open(manifest_path, "w"))
    print(f"[{year}] done: {len(manifest['done'])} extracted, {len(manifest['failed'])} failed")


if __name__ == "__main__":
    for year in resolve_years(scraper["years"]):
        try:
            scrape_year(year)
        except Exception as error:
            print(f"[{year}] failed: {error}")
