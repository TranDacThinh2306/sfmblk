# Weights & Biases (W&B) Setup & Monitoring Guide

## 📊 Overview

Đã thêm **Weights & Biases** để giám sát training losses (ce_loss, lb_loss, z_loss) real-time.

## 🚀 Installation

```bash
pip install wandb
```

## 🔐 Authentication

### Login vào W&B

```bash
wandb login
```

Lệnh này sẽ:
1. Mở browser yêu cầu đăng nhập
2. Copy API key
3. Paste vào terminal

**Hoặc** set environment variable:
```bash
export WANDB_API_KEY=your_api_key_here
```

## 📈 Monitored Metrics

Training sẽ log các metrics sau:

| Metric | Mô tả |
|--------|-------|
| `total_loss` | `ce_loss + w_lb*lb_loss + w_z*z_loss` |
| `ce_loss` | Cross-entropy loss (LLM prediction) |
| `lb_loss` | Load-balance loss (expert balancing) |
| `z_loss` | Z-loss (auxiliary loss) |
| `learning_rate` | Current learning rate |
| `train/loss` | Training loss per step |
| `eval/ngram_1` | 1-gram precision |
| `eval/ngram_2` | 2-gram precision |
| `eval/ngram_3` | 3-gram precision |
| `eval/exact_match` | Exact match score |

## ▶️ Running Training with W&B

### Tự động (scripts)
```bash
# Linux/Mac
bash run_training.sh 4

# Windows
.\run_training.bat 4
```

Scripts đã được cấu hình với `--report_to wandb tensorboard`.

### Thủ công
```bash
torchrun --nproc_per_node=4 vintern_moe_pretrain.py \
    --model_name "5CD-AI/Vintern-1B-v3_5" \
    --meta_path "./data/meta.json" \
    --output_dir "./outputs/vintern_moe" \
    --report_to wandb \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3
```

## 🎯 Viewing Results

### 1. Web Dashboard
Truy cập: https://wandb.ai/your-username/vintern-moe-training

### 2. Command Line
```bash
# View latest run
wandb sync --sync-all

# View specific run
wandb pull your-username/vintern-moe-training/run_id
```

### 3. Local Offline Mode (tùy chọn)
```bash
# Chạy offline, sync sau
export WANDB_MODE=offline

# Khi online, sync data
wandb sync ./wandb/offline-run-xxxxx
```

## 📊 Understanding Loss Components

### Ce Loss (Cross-Entropy)
- **Mục tiêu**: LLM dự đoán đúng token kế tiếp
- **Kỳ vọng**: Giảm dần qua epochs
- **Ví dụ**: 2.5 → 1.8 → 1.2

### LB Loss (Load-Balance)
- **Mục tiêu**: Phân phối dữ liệu đều giữa experts
- **Công thức**: Prevent experts từ collapse
- **Kỳ vọng**: Nhỏ, thường < 0.1
- **Tuning**: Tăng `w_lb` nếu experts imbalanced

### Z Loss (Auxiliary)
- **Mục tiêu**: Regularization để ổn định training
- **Kỳ vọng**: Rất nhỏ, thường < 0.001
- **Tuning**: Điều chỉnh `w_z` parameter

## 🔧 Customizing Logging

### Thay đổi Project Name
```bash
# Trong code hoặc command line
# Sửa tên trong wandb.init()
```

### Thêm Tags
```bash
# Trong vintern_moe_pretrain.py, MoETrainer.log()
# Hoặc dùng W&B dashboard UI
```

## 📉 Troubleshooting

### Lỗi: "wandb not installed"
```bash
pip install wandb
```

### Lỗi: "Not logged in"
```bash
wandb login
```

### Lỗi: "Run already exists"
```bash
# Xóa .wandb cache hoặc dùng tên run khác
rm -rf .wandb
```

### Chạy offline (no internet)
```bash
export WANDB_MODE=offline
# Training sẽ lưu local, sync khi có internet
wandb sync
```

## 💡 Best Practices

1. **Run Name Descriptive**: Tự động tạo từ config
   ```
   moe_lr0.0002_bs4  ← Dễ track
   ```

2. **Tag cho Organization**:
   ```
   tags=["moe", "vintern", "vqa", "v1.0"]
   ```

3. **Monitor Early Stopping**:
   - Nếu `total_loss` không giảm, dừng early
   - Check `eval/exact_match` tiến độ

4. **Compare Runs**:
   - Dùng W&B dashboard để so sánh multiple runs
   - Export data dạng CSV

## 📋 Example: Monitoring Loss Ratios

Nên monitor ratio giữa losses:

```
total_loss = ce_loss + 0.01*lb_loss + 0.001*z_loss

Mục tiêu:
- ce_loss: 60-80% of total_loss
- lb_loss: 15-30% of total_loss
- z_loss: 1-5% of total_loss
```

## 🔄 Integration với Training Loop

**Đã tự động log:**
- ✅ `ce_loss`, `lb_loss`, `z_loss` mỗi logging step
- ✅ `eval/ngram_*` metrics mỗi eval step
- ✅ Learning rate changes
- ✅ Training/eval loss

**Custom logging** (nếu cần):
```python
# Trong MoETrainer.log()
if WANDB_AVAILABLE and wandb.run is not None:
    wandb.log({
        "custom_metric": value,
    }, step=self.state.global_step)
```

## 📚 Resources

- W&B Docs: https://docs.wandb.ai
- HuggingFace Integration: https://docs.wandb.ai/guides/integrations/huggingface
- Debugging Guide: https://docs.wandb.ai/guides/technical-faq/general
