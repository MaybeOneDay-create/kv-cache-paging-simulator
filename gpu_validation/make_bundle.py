"""Package source and a numeric trace for a Colab upload; omit raw dialogue."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "review_outputs" / "gpu_validation_bundle.zip"
FILES = [
    *sorted((ROOT / "gpu_validation").glob("*.py")),
    ROOT / "gpu_validation" / "Colab快速开始.md",
    *sorted((ROOT / "kv_cache").glob("*.py")),
    ROOT / "run_trace_experiment.py",
    ROOT / "review_outputs" / "sharegpt" / "trace.json",
    ROOT / "docs" / "云端GPU验证记录模板.md",
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
        for source in FILES:
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, source.relative_to(ROOT).as_posix())
    print(OUTPUT)


if __name__ == "__main__":
    main()
