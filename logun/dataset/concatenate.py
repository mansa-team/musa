from pathlib import Path

import yaml

config_path = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
paths = config["paths"]


def concatenate():
    output_path = Path(paths["corpus"])
    files = sorted(Path(paths["output_dir"]).glob("*.txt"))
    if not files:
        print(f"No files found in {paths['output_dir']}")
        return

    with open(output_path, "w", encoding="utf-8") as out_file:
        for file in files:
            content = file.read_text(encoding="utf-8")
            out_file.write(content)
            out_file.write("\n")

    print(f"{len(files)} files -> {output_path}")


if __name__ == "__main__":
    concatenate()
