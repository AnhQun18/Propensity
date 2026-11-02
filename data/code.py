from pathlib import Path

import kagglehub


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "ecommerce"
    print("Downloading Kaggle dataset to data/ecommerce/", flush=True)
    kagglehub.dataset_download(
        "mkechinov/ecommerce-behavior-data-from-multi-category-store",
        output_dir=str(output_dir),
    )
    print("Download complete: data/ecommerce/", flush=True)


if __name__ == "__main__":
    main()
