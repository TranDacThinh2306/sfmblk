# Hướng dẫn Chạy Vintern MoE Pretraining với Torch Distributed

## 📋 Yêu cầu

- PyTorch >= 1.13
- Transformers >= 4.30
- Một hoặc nhiều GPU (CUDA)
- `petrel_client` (tùy chọn, cho loading ảnh từ S3)

## 🚀 Cài đặt

```bash
# Cài đặt dependencies
pip install torch transformers datasets pillow torchvision
```

## 📁 Cấu trúc Dữ liệu

Chuẩn bị file meta JSONL:

```bash
mkdir -p ./data
# Tạo file meta.json với định dạng:
# {
#     "dataset_name": {
#         "annotation": "path/to/dataset.jsonl",
#         "root": "path/to/images/",
#         "repeat_time": 1,
#         "data_augment": true
#     }
# }
```

Định dạng JSONL cho mỗi dòng:
```json
{
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nCâu hỏi của người dùng"},
    {"from": "gpt", "value": "Câu trả lời"}
  ]
}
```

## 🎯 Chạy Training

### Linux/Mac (Bash):
```bash
# Tự động phát hiện số GPU
bash run_training.sh

# Hoặc chỉ định số GPU
bash run_training.sh 4
```

### Windows (PowerShell):
```powershell
# Tự động phát hiện số GPU
.\run_training.bat

# Hoặc chỉ định số GPU
.\run_training.bat 4
```

### Windows (Batch):
```cmd
REM Tự động phát hiện số GPU
run_training.bat

REM Hoặc chỉ định số GPU
run_training.bat 4
```

### Chạy thủ công với torchrun:
```bash
# Single GPU
python vintern_moe_pretrain.py \
    --model_name "5CD-AI/Vintern-1B-v3_5" \
    --meta_path "./data/meta.json" \
    --output_dir "./outputs/vintern_moe" \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 vintern_moe_pretrain.py \
    --model_name "5CD-AI/Vintern-1B-v3_5" \
    --meta_path "./data/meta.json" \
    --output_dir "./outputs/vintern_moe" \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3
```

## ⚙️ Các Tham số Quan Trọng

### Model Arguments
| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `model_name` | `5CD-AI/Vintern-1B-v3_5` | Tên mô hình HuggingFace |
| `num_experts` | 8 | Số experts trong MoE |
| `top_k` | 2 | Số experts được chọn |
| `num_shared` | 1 | Số shared experts |
| `w_lb` | 0.01 | Load-balance loss weight |
| `w_z` | 0.001 | Z-loss weight |

### Freeze Options (Chỉ train MoE)
| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `freeze_vision` | True | Freeze vision encoder |
| `freeze_llm` | True | Freeze language model |
| `freeze_mlp1` | True | Freeze original mlp1 |

### Data Arguments
| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `meta_path` | None | **Bắt buộc** - Đường dẫn file meta.json |
| `max_seq_length` | 2048 | Độ dài max sequence |
| `force_image_size` | 448 | Kích thước ảnh |
| `dynamic_image_size` | False | Dùng dynamic image patches |
| `ngram_n` | [1,2,3] | N-gram sizes cho metrics |

### Training Arguments
| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `learning_rate` | 2e-4 | Learning rate |
| `num_train_epochs` | 3 | Số epochs training |
| `output_dir` | `./outputs/vintern_moe_training` | Thư mục lưu checkpoint |
| `save_steps` | 500 | Lưu checkpoint mỗi N steps |
| `logging_steps` | 10 | Log mỗi N steps |

## 📊 Monitoring

Training logs được lưu trong `output_dir`:
```
outputs/vintern_moe_training/
├── runs/                    # TensorBoard logs
├── checkpoint-500/          # Saved checkpoints
├── moe_weights.pt          # Chỉ MoE weights
└── trainer_state.json      # Training state
```

Xem TensorBoard:
```bash
tensorboard --logdir outputs/vintern_moe_training/runs
```

## 🔧 Tùy Chỉnh Script

Chỉnh sửa các biến trong script `run_training.sh` hoặc `run_training.bat`:

```bash
# Ví dụ: Thay đổi batch size
PER_DEVICE_TRAIN_BATCH_SIZE=8

# Ví dụ: Thay đổi learning rate
LEARNING_RATE=1e-4

# Ví dụ: Training lâu hơn
NUM_TRAIN_EPOCHS=10
```

## ⚡ Tối ưu Hiệu Suất

### Để tăng tốc độ:
1. Tăng `per_device_train_batch_size` (nếu memory cho phép)
2. Tăng `gradient_accumulation_steps` để tăng effective batch size
3. Dùng `dynamic_image_size=True` nếu ảnh có kích thước khác nhau

### Để tiết kiệm memory:
1. Giảm `per_device_train_batch_size`
2. Dùng `gradient_checkpointing=True`
3. Giảm `max_seq_length`

## 🐛 Troubleshooting

### Lỗi: "meta_path must be specified"
```bash
# Chắc chắn rằng meta_path được set đúng:
--meta_path "./data/meta.json"
```

### Lỗi: "CUDA out of memory"
```bash
# Giảm batch size
--per_device_train_batch_size 2

# Hoặc tăng gradient accumulation
--gradient_accumulation_steps 8
```

### Lỗi: "No module named 'petrel_client'"
```bash
# Bỏ qua (sử dụng PIL thay thế):
pip install petrel_client  # Hoặc bỏ qua
```

## 💾 Resume Training

```bash
torchrun --nproc_per_node=4 vintern_moe_pretrain.py \
    --model_name "5CD-AI/Vintern-1B-v3_5" \
    --output_dir "./outputs/vintern_moe" \
    --resume_from_checkpoint "./outputs/vintern_moe/checkpoint-500"
```

## 📝 Notes

- **MoE Weight**: Chỉ MoE module được train, vision + LLM được freeze
- **Loss**: `total_loss = ce_loss + w_lb * lb_loss + w_z * z_loss`
- **Query Embedding**: Tự động extract từ question tokens
- **Metrics**: N-gram precision + exact match được tính lúc evaluate
