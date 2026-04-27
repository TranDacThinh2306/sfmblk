import argparse
import ast
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def str_to_answer_list(answers):
    """Parse answers using ast.literal_eval when possible."""
    try:
        parsed = ast.literal_eval(str(answers))
        return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        return [str(answers).strip()]


def build_autovqa_jsonl(csv_path, images_root, output_jsonl, num_samples):
    old_prefix = "http://images.cocodataset.org/train2017/"
    images_root = str(images_root).rstrip("/") + "/"

    df = pd.read_csv(csv_path)[["question", "answers", "image_link"]]
    df["image"] = df["image_link"].apply(
        lambda x: images_root + x.split("/")[-1]
        if isinstance(x, str) and ("cocodataset" in x or x.startswith(old_prefix))
        else x
    )

    processed = []
    total_rows = min(num_samples, len(df))

    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Preprocessing"):
        if len(processed) >= num_samples:
            break

        try:
            with Image.open(row["image"]) as img:
                width, height = img.size
        except Exception:
            width, height = 0, 0

        answers = str_to_answer_list(row["answers"])
        chosen_answer = random.choice(answers) if answers else ""

        processed.append(
            {
                "id": f"vivqa_{idx:06d}",
                "image": row["image"],
                "width": width,
                "height": height,
                "question": row["question"],
                "chosen_answer": chosen_answer,
            }
        )

    all_data = [
        {
            "id": row["id"],
            "image": row["image"],
            "width": row["width"],
            "height": row["height"],
            "conversations": [
                {"from": "human", "value": f"<image>\\n{row['question']}"},
                {"from": "gpt", "value": row["chosen_answer"]},
            ],
        }
        for row in processed
    ]

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done: created {len(all_data)} samples at {output_jsonl}")
    return len(all_data)


def write_metadata_file(images_root, annotation_file, metadata_path, dataset_len):
    metadata = {
        "autovqa_vn": {
            "root": str(images_root),
            "annotation": str(annotation_file),
            "data_augment": False,
            "repeat_time": 1,
            "length": dataset_len,
        }
    }

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Done: metadata saved to {metadata_path}")

def download_pretrained_model(pretrained_dir, model_repo, model_local_dir):
    pretrained_dir = Path(pretrained_dir)
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "huggingface-cli",
            "download",
            "--resume-download",
            "--local-dir-use-symlinks",
            "False",
            model_repo,
            "--local-dir",
            model_local_dir,
        ],
        cwd=str(pretrained_dir),
    )

def run_command(command, cwd=None):
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def write_train_script(
    train_script_path,
    model_name,
    meta_path,
    output_dir,
    num_gpus=1,
    num_epochs=5,
    per_device_batch_size=4,
    learning_rate=2e-4,
):
        script_text = f"""#!/bin/bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


echo "PYTHONPATH set: $PYTHONPATH"

MODEL_NAME="{model_name}"
META_PATH="{meta_path}"
OUTPUT_DIR="{output_dir}"
NUM_GPUS={num_gpus}

torchrun --nproc_per_node=$NUM_GPUS --master_port=12345 \\
    vintern_moe_pretrain.py \\
    --model_name "$MODEL_NAME" \\
    --output_dir "$OUTPUT_DIR" \\
    --meta_path "$META_PATH" \\
    --conv_style internvl2_5 \\
    --max_seq_length 2048 \\
    --dynamic_image_size True \\
    --use_thumbnail True \\
    --min_dynamic_patch 1 \\
    --max_dynamic_patch 6 \\
    --down_sample_ratio 0.5 \\
    --force_image_size 448 \\
    --num_train_epochs {num_epochs} \\
    --per_device_train_batch_size {per_device_batch_size} \\
    --gradient_accumulation_steps 2 \\
    --learning_rate {learning_rate} \\
    --weight_decay 0.01 \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type cosine \\
    --max_grad_norm 1.0 \\
    --bf16 True \\
    --dataloader_num_workers 4 \\
    --save_strategy steps \\
    --save_steps 500 \\
    --save_total_limit 3 \\
    --logging_steps 10 \\
    --report_to wandb tensorboard \\
    --remove_unused_columns False
"""

        train_script_path = Path(train_script_path)
        train_script_path.parent.mkdir(parents=True, exist_ok=True)
        train_script_path.write_text(script_text, encoding="utf-8")

        # Best effort for Linux/Kaggle; harmless on other systems.
        try:
            os.chmod(train_script_path, 0o755)
        except OSError:
            pass

        print(f"Done: training script saved to {train_script_path}")


def run_training_script(train_script_path):
    run_command(["bash", str(train_script_path)])


def check_transformers_and_tokenizer(model_path):
    run_command([sys.executable, "-m", "pip", "show", "transformers"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_text = "Hello world"
    ids = tokenizer(test_text).input_ids

    print(f"Token ids: {ids}")
    print(f"First token is BOS: {ids[0] == tokenizer.bos_token_id}")
    print(f"tokenizer.legacy: {getattr(tokenizer, 'legacy', 'N/A')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vinternmoe notebook flow from cell 4 onward."
    )
    parser.add_argument(
        "--csv-path",
        default="/kaggle/input/datasets/nguynrichard/auto-vivqa/text/text/evaluate_60k_data_balanced_preprocessed_train_temp.csv",
    )
    parser.add_argument(
        "--images-root",
        default="/kaggle/input/datasets/nguynrichard/auto-vivqa/images/images",
    )
    parser.add_argument("--output-jsonl", default="/kaggle/working/autovqa_vn.jsonl")
    parser.add_argument(
        "--metadata-path", default="/kaggle/working/finetune_custom_autovqavn.json"
    )
    parser.add_argument("--num-samples", type=int, default=10000)

    # parser.add_argument(
    #     "--internvl-source",
    #     default="/kaggle/input/datasets/chalicetrncthnh/moevintern",
    # )
    parser.add_argument("--model-name", default="5CD-AI/Vintern-1B-v3_5")
    parser.add_argument("--train-script", default="/kaggle/working/train_moe.sh")
    parser.add_argument("--output-dir", default="/kaggle/working/vintern_moe_finetune")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per-device-batch-size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")

    parser.add_argument(
        "--skip-tokenizer-check",
        action="store_true",
        help="Skip tokenizer sanity check",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip launching training script"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ════════════════════════════════════════════════════════════════════════════════
    # Step 1: Build dataset and metadata
    # ════════════════════════════════════════════════════════════════════════════════
    dataset_len = build_autovqa_jsonl(
        csv_path=args.csv_path,
        images_root=args.images_root,
        output_jsonl=args.output_jsonl,
        num_samples=args.num_samples,
    )

    write_metadata_file(
        images_root=args.images_root,
        annotation_file=args.output_jsonl,
        metadata_path=args.metadata_path,
        dataset_len=dataset_len,
    )

    # ════════════════════════════════════════════════════════════════════════════════
    # Step 2: Generate training script
    # ════════════════════════════════════════════════════════════════════════════════
    write_train_script(
        train_script_path=args.train_script,
        model_name=args.model_name,
        meta_path=args.metadata_path,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
    )

    # ════════════════════════════════════════════════════════════════════════════════
    # Step 3: Run training
    # ════════════════════════════════════════════════════════════════════════════════
    if not args.skip_train:
        run_training_script(args.train_script)

    # ════════════════════════════════════════════════════════════════════════════════
    # Step 4: Verify tokenizer (optional)
    # ════════════════════════════════════════════════════════════════════════════════
    if not args.skip_tokenizer_check:
        check_transformers_and_tokenizer(args.model_name)


if __name__ == "__main__":
    main()
