import kagglehub
import os


def fetch_kaggle_dataset(dataset_name, output_path):
    # Tải dataset từ Kaggle
    terminal_cwd = os.getcwd()
    final_output_path = os.path.join(terminal_cwd, output_path.lstrip("/\\"))
    os.makedirs(final_output_path, exist_ok=True)

    path = kagglehub.dataset_download(handle=dataset_name, output_dir=final_output_path)
    print(f"Downloaded {dataset_name} -> {path}")


def main():
    terminal_cwd = os.getcwd()
    os.makedirs(os.path.join(terminal_cwd, "kaggle", "input"), exist_ok=True)
    os.makedirs(os.path.join(terminal_cwd, "kaggle", "working"), exist_ok=True)

    fetch_kaggle_dataset("nguynrichard/auto-vivqa", "kaggle/input/datasets/nguynrichard/auto-vivqa")
    # fetch_kaggle_dataset("chalicetrncthnh/moevintern", "kaggle/input/datasets/chalicetrncthnh/moevintern")


if __name__ == "__main__":
    main()