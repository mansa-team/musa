import json
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
paths = config["paths"]

def concatenate():
    output_path = Path(paths["corpus"])
    files = sorted(Path(paths["output_dir"]).glob("*.txt"))
    if not files:
        print(f"No files found in {paths['output_dir']}")
        return

    total_bytes = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for file in files:
            content = file.read_text(encoding="utf-8")
            total_bytes += len(content)
            out.write(content)
            out.write("\n")
            print(f"  {file.name}: {len(content):,} chars")

    print(f"{len(files)} files → {output_path} ({total_bytes:,} chars, ~{total_bytes // 4:,} tokens)")


if __name__ == "__main__":
    concatenate()
