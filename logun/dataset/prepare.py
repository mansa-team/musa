import json, re, hashlib
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
dapt = config["dapt"]
paths = config["paths"]

CLEANING_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:FOLHA|PAG(?:INA|É?)?)\s*\d+\s*/\s*\d+\s*(?:\n|$)", re.I),
    re.compile(r"(?:^|\n)\s*(?:Página\s+\d+\s+de\s+\d+|Page\s+\d+\s+of\s+\d+)\s*(?:\n|$)", re.I),
    re.compile(r"CNPJ[:\s]*\d[\d.\-\/]{17,}\s*(?:\n|$)", re.I),
    re.compile(r"(?:^|\n)\s*(?:[A-ZÁÉÍÓÚÃÕÊÔ][A-ZÁÉÍÓÚÃÕÊÔ\s]{10,})\s*(?:CNPJ|cnpj)\s*[:\-]?\s*[\d.\-\/]+", re.I),
    re.compile(r"(?:^|\n)\s*(?:O\s+conteúdo\s+deste\s+documento|Declaração\s+de\s+responsabilidade|Este\s+documento\s+(?:foi|contém)\s+objetos?\s+de\s+(?:catequese|transcrição))\s.*?(?:\n\s*\n|\Z)", re.I | re.DOTALL),
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    re.compile(r"\n{3,}"),
]
PORTUGUESE_WORDS = {
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem", "sob", "sobre", "que", "como", "mais",
    "ou", "e", "a", "o", "as", "os", "um", "uma", "ser", "estar", "ter",
    "haver", "fazer", "poder", "dizer", "não", "também", "já", "ainda",
    "muito", "todo", "cada", "quando", "onde", "porque", "então", "assim",
    "até", "desde", "durante", "após", "antes",
}
PORTUGUESE_ACCENTS = re.compile(r"[áéíóúâêôãõçàÁÉÍÓÚÂÊÔÃÕÇÀ]")


def clean_text(text):
    for pattern in CLEANING_PATTERNS:
        if pattern.flags & re.DOTALL:
            text = pattern.sub("", text)
        else:
            text = pattern.sub("\n", text) if "FOLHA" in pattern.pattern or "Página" in pattern.pattern or "CNPJ" in pattern.pattern else pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_documents(documents):
    seen_hashes = set()
    exact_deduped = []
    for document in documents:
        text_hash = hash_text(document["text"])
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            exact_deduped.append(document)

    seen_paragraphs = set()
    result = []
    for document in exact_deduped:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document["text"]) if p.strip() and len(p.strip()) >= 50]
        filtered_paragraphs = []
        for paragraph in paragraphs:
            paragraph_hash = hash_text(paragraph)
            if paragraph_hash not in seen_paragraphs:
                seen_paragraphs.add(paragraph_hash)
                filtered_paragraphs.append(paragraph)
        if filtered_paragraphs:
            result.append({**document, "text": "\n\n".join(filtered_paragraphs)})
    return result


def chunk_text(text, chunk_size=1800, overlap=200):
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current_chunk = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > chunk_size:
            chunks.append(current_chunk)
            current_chunk = (current_chunk[-overlap:] + "\n\n" + paragraph) if overlap and len(current_chunk) > overlap else paragraph
        else:
            current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
        while len(current_chunk) > chunk_size * 1.5:
            split_position = current_chunk.rfind(" ", 0, chunk_size)
            if split_position < chunk_size // 2:
                split_position = chunk_size
            chunks.append(current_chunk[:split_position].strip())
            current_chunk = current_chunk[split_position:].strip()
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def is_portuguese(text):
    if not text or len(PORTUGUESE_ACCENTS.findall(text)) < 3:
        return False
    words = text.lower().split()[:200]
    portuguese_count = sum(1 for word in words if word in PORTUGUESE_WORDS)
    return portuguese_count / max(len(words), 1) > 0.05


def passes_quality_filter(chunk, min_length=200, max_length=8000):
    if not (min_length <= len(chunk) <= max_length):
        return False
    if not is_portuguese(chunk):
        return False
    digit_ratio = sum(character.isdigit() for character in chunk) / max(len(chunk), 1)
    return digit_ratio <= 0.7


def format_output(chunks, output_dir, stem):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    text_content = "\n".join(chunk.replace("\n", " ") for chunk in chunks)
    (output_path / f"{stem}.txt").write_text(text_content, encoding="utf-8")
    jsonl_content = "\n".join(json.dumps({"text": chunk}, ensure_ascii=False) for chunk in chunks)
    (output_path / f"{stem}.jsonl").write_text(jsonl_content, encoding="utf-8")


def run_pipeline(config):
    documents = []
    with open(paths["input"], encoding="utf-8") as file:
        for line in tqdm(file, desc="Loading"):
            if line.strip():
                documents.append(json.loads(line))

    total_characters = sum(len(document.get("text", "")) for document in documents)
    print(f"{len(documents)} documents, {total_characters:,} characters")

    for document in documents:
        document["text"] = clean_text(document.get("text", ""))

    documents = deduplicate_documents(documents)
    print(f"After deduplication: {len(documents)} documents")

    all_chunks = [chunk for document in documents for chunk in chunk_text(document["text"], dapt["chunk_size"], dapt["overlap"])]
    print(f"Total chunks: {len(all_chunks):,}")

    seen_hashes = set()
    filtered_chunks = []
    for chunk in all_chunks:
        chunk_hash = hash_text(chunk)
        if chunk_hash not in seen_hashes and passes_quality_filter(chunk, dapt["min_length"], dapt["max_length"]):
            seen_hashes.add(chunk_hash)
            filtered_chunks.append(chunk)

    estimated_tokens = sum(len(chunk) for chunk in filtered_chunks) // 4
    print(f"After filtering: {len(filtered_chunks):,} chunks ({estimated_tokens:,} estimated tokens)")

    if dapt["dry_run"]:
        return

    stem = Path(paths["input"]).stem
    format_output(filtered_chunks, paths["output_dir"], stem)
    print(f"Output written to: {paths['output_dir']}/{stem}.*")


if __name__ == "__main__":
    run_pipeline(paths)
