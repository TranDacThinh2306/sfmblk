"""
TextVQA fine-tuning với HuggingFace Trainer framework
- Chèn QueryConditionedMoE sau mlp1
- Custom Trainer xử lý total_loss = ce + lb + z
- compute_metrics tính n-gram accuracy lúc evaluate
"""
import gc
import os
import re
import json
import math
import random
import traceback
import torch
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass, field
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import logging
import sys
import warnings
import numpy as np

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvalPrediction
from torch.utils.data import Dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
# ... các import khác ...
import sys

# ── Auto-add thư mục chứa file này vào sys.path ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modeling_internvl_moe import QueryConditionedMoE, get_query_emb
from dataset import (ConcatDataset, TCSLoader,
                        WeightedConcatDataset, build_transform,
                        dynamic_preprocess, preprocess,
                        preprocess_internlm)

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# ════════════════════════════════════════════════════════════════════════════════
# Utils. Patch
# ════════════════════════════════════════════════════════════════════════════════
def concat_pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            tensors = [
                f[k] if isinstance(f[k], torch.Tensor)
                else torch.tensor(f[k])
                for f in features
            ]
            batch[k] = torch.cat(tensors)
            if k == 'image_flags':
                batch[k] = batch[k].long()
    return batch


# ════════════════════════════════════════════════════════════════════════════════
# 1. Arguments
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelArguments:
    model_name: str = field(
        default="5CD-AI/Vintern-1B-v3_5",
        metadata={"help": "HuggingFace model id hoặc local path"},
    )
    vis_hidden: int  = field(default=896)
    lm_hidden: int   = field(default=896)
    num_experts: int = field(default=4)
    top_k: int       = field(default=2)
    num_shared: int  = field(default=1)
    moe_dropout: float = field(default=0.0)
    
    # ═════════════════════════════════════════
    # Freeze options (freeze everything except MoE by default)
    # ═════════════════════════════════════════
    freeze_vision: bool = field(
        default=True,
        metadata={"help": "Freeze vision encoder (InternVIT). Default: True"}
    )
    freeze_llm: bool = field(
        default=True,
        metadata={"help": "Freeze language model (LLM). Default: True"}
    )
    freeze_mlp1: bool = field(
        default=True,
        metadata={"help": "Freeze original mlp1 (only MoE will be trainable). Default: True"}
    )
    
    w_lb: float      = field(default=0.01,  metadata={"help": "load-balance loss weight"})
    w_z: float       = field(default=0.001, metadata={"help": "z-loss weight"})


@dataclass
class DataArguments:
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={'help': 'The maximum total input sequence length after tokenization.'},
    )
    force_image_size: Optional[int] = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internvl2_5', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )
    max_new_tokens: int = field(default=64, metadata={'help': 'Max tokens to generate during inference.'})
    ngram_n: List[int]  = field(default_factory=lambda: [1, 2, 3], metadata={'help': 'N-gram sizes for evaluation.'})


# ════════════════════════════════════════════════════════════════════════════════
# 2. MLP1 Wrapper
# ════════════════════════════════════════════════════════════════════════════════
class MLP1WithMoE(nn.Module):
    """
    Wrap mlp1 gốc + QueryConditionedMoE phía sau.
    - Trả về tensor đơn → InternVLChatModel.forward không cần sửa.
    - lb_loss / z_loss lưu vào attribute để Trainer lấy ra.
    - query_emb được inject từ bên ngoài trước mỗi forward call.
    """
    def __init__(self, original_mlp1: nn.Module, moe: QueryConditionedMoE):
        super().__init__()
        self.mlp1     = original_mlp1
        self.moe      = moe
        self._query_emb: Optional[torch.Tensor] = None
        self.lb_loss  = torch.tensor(0.0)
        self.z_loss   = torch.tensor(0.0)

    def set_query_emb(self, query_emb: torch.Tensor):
        self._query_emb = query_emb

    def forward(self, vis_emb: torch.Tensor) -> torch.Tensor:
        vis_emb              = self.mlp1(vis_emb)
        vis_emb, lb, z       = self.moe(vis_emb, self._query_emb)
        self.lb_loss         = lb
        self.z_loss          = z
        return vis_emb


# ════════════════════════════════════════════════════════════════════════════════
# 3. Image transform
# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# 4. Dataset
# ════════════════════════════════════════════════════════════════════════════════

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=224,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        repeat_time=1,
        normalize_type='imagenet',
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'internlm2-chat' or self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internlm
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, 
                                    input_size=self.image_size,
                                    pad2square=self.pad2square,
                                    normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Extract question from first human message (for query embedding)
        question = data_item['conversations'][0]['value'].replace('<image>\n', '').strip()

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Tokenize question separately for query embedding
        q_enc = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128,
        )

        # Extract answer from second message (if exists)
        answer = data_item['conversations'][1]['value'] if len(data_item['conversations']) > 1 else ''

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            q_input_ids=q_enc['input_ids'].squeeze(0),
            q_attention_mask=q_enc['attention_mask'].squeeze(0),
            question=question,
            answer=answer,
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=self.max_dynamic_patch // num_image,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Extract question and answer from conversations
        question = data_item['conversations'][0]['value'].strip() if len(data_item['conversations']) > 0 else ''
        answer = data_item['conversations'][1]['value'] if len(data_item['conversations']) > 1 else ''

        # Tokenize question separately for query embedding
        q_enc = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128,
        )

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_image)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            q_input_ids=q_enc['input_ids'].squeeze(0),
            q_attention_mask=q_enc['attention_mask'].squeeze(0),
            question=question,
            answer=answer,
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Extract question and answer from conversations
        question = data_item['conversations'][0]['value'].strip() if len(data_item['conversations']) > 0 else ''
        answer = data_item['conversations'][1]['value'] if len(data_item['conversations']) > 1 else ''

        # Tokenize question separately for query embedding
        q_enc = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128,
        )

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            q_input_ids=q_enc['input_ids'].squeeze(0),
            q_attention_mask=q_enc['attention_mask'].squeeze(0),
            question=question,
            answer=answer,
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                print(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset




# ════════════════════════════════════════════════════════════════════════════════
# 5. Data collator
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class VQADataCollator:
    """
    Trainer yêu cầu collator trả về dict tensor.
    Strings (question, answer) được giữ lại để dùng trong compute_metrics.
    """
    tokenizer: Any
    dtype: torch.dtype = torch.bfloat16

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        # Tensor fields
        tensor_keys = [
            "pixel_values", "input_ids", "attention_mask",
            "labels", "q_input_ids", "q_attention_mask", "image_flags"
        ]
        for k in tensor_keys:
            if k in features[0]:  # Only include if present in data
                batch[k] = torch.stack([f[k] for f in features])

        # pixel_values cần đúng dtype
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(self.dtype)

        # String fields — giữ dưới dạng list
        if "question" in features[0]:
            batch["question"] = [f["question"] for f in features]
        if "answer" in features[0]:
            batch["answer"]   = [f["answer"]   for f in features]

        return batch


# ════════════════════════════════════════════════════════════════════════════════
# 6. N-gram metrics
# ════════════════════════════════════════════════════════════════════════════════
def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())


def ngram_precision(pred: str, ref: str, n: int) -> float:
    p_tok = normalize_text(pred).split()
    r_tok = normalize_text(ref).split()
    if len(p_tok) < n:
        return 0.0
    p_ng = Counter(tuple(p_tok[i:i+n]) for i in range(len(p_tok) - n + 1))
    r_ng = Counter(tuple(r_tok[i:i+n]) for i in range(len(r_tok) - n + 1))
    overlap = sum((p_ng & r_ng).values())
    total   = sum(p_ng.values())
    return overlap / total if total else 0.0


def compute_ngram_metrics(preds: List[str], refs: List[str], ns: List[int]) -> Dict[str, float]:
    out = {}
    for n in ns:
        scores = [ngram_precision(p, r, n) for p, r in zip(preds, refs)]
        out[f"ngram_{n}"] = sum(scores) / len(scores) if scores else 0.0
    exact = [1.0 if normalize_text(p) == normalize_text(r) else 0.0 for p, r in zip(preds, refs)]
    out["exact_match"] = sum(exact) / len(exact) if exact else 0.0
    return out


# ════════════════════════════════════════════════════════════════════════════════
# 7. Custom Trainer
# ════════════════════════════════════════════════════════════════════════════════
class MoETrainer(Trainer):
    """
    Override compute_loss để:
      1. Inject query_emb vào mlp1 wrapper trước khi model.forward()
      2. Cộng lb_loss và z_loss vào ce_loss
    Override prediction_step để generate và tính n-gram.
    """

    def __init__(
        self,
        *args,
        model_args: ModelArguments,
        data_args: DataArguments,
        tokenizer,
        **kwargs,
    ):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.model_args = model_args
        self.data_args  = data_args
        self._tokenizer = tokenizer

    def _unwrap(self,model):
        # Unwrap model if it's wrapped in DataParallel or DistributedDataParallel
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module
        return model

    # ── Inject query_emb và tính total_loss ────────────────────────────────────
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        q_input_ids      = inputs.pop("q_input_ids")
        q_attention_mask = inputs.pop("q_attention_mask")
        logger.warning(q_input_ids.shape)

        model = self._unwrap(model)  # <-- unwrap
        # Lấy query_emb từ question tokens (no_grad vì LM bị frozen)
        with torch.no_grad():
            query_emb = get_query_emb(
                language_model = model.language_model.model,
                input_ids      = q_input_ids,
                attention_mask = q_attention_mask,
            )                                                   # [B, lm_hidden]

        # Inject vào wrapper
        model.mlp1.set_query_emb(query_emb)

        # Forward bình thường
        outputs  = model(**inputs)
        ce_loss  = outputs.loss
        lb_loss  = model.mlp1.lb_loss.to(ce_loss.device)
        z_loss   = model.mlp1.z_loss.to(ce_loss.device)

        total_loss = (
            ce_loss
            + self.model_args.w_lb * lb_loss
            + self.model_args.w_z  * z_loss
        )

        # Log auxiliary losses vào Trainer
        self.log({
            "ce_loss": ce_loss.detach().item(),
            "lb_loss": lb_loss.detach().item(),
            "z_loss" : z_loss.detach().item(),
        })

        return (total_loss, outputs) if return_outputs else total_loss

    # ── Generate để compute_metrics dùng ───────────────────────────────────────
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        # ignore_keys=None,
    ):
        questions = inputs.pop("question", None)
        answers   = inputs.pop("answer",   None)

        q_input_ids      = inputs.pop("q_input_ids")
        q_attention_mask = inputs.pop("q_attention_mask")
        pixel_values     = inputs["pixel_values"]
        with torch.no_grad():
            query_emb = get_query_emb(
                language_model = model.language_model.model,
                input_ids      = q_input_ids,
                attention_mask = q_attention_mask,
            )
            model.mlp1.set_query_emb(query_emb)

            # ── Tính loss ──────────────────────────────────────────────────────
            outputs  = model(**inputs)
            ce_loss  = outputs.loss
            lb_loss  = model.mlp1.lb_loss.to(ce_loss.device)
            z_loss   = model.mlp1.z_loss.to(ce_loss.device)
            loss     = ce_loss + self.model_args.w_lb * lb_loss + self.model_args.w_z * z_loss

            if prediction_loss_only or questions is None:
                logger.warning("[Question] question is empty, skip generation and metric computation.")
                return (loss, None, None)

            # ── Generate câu trả lời ───────────────────────────────────────────
            prompts = [
                f"<image>\nQuestion: {q}\nAnswer:" for q in questions
            ]
            prompt_enc = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(model.device)

            model.mlp1.set_query_emb(query_emb)   # set lại vì generate gọi forward nhiều lần

            gen_ids = model.generate(
                pixel_values   = pixel_values,
                input_ids      = prompt_enc["input_ids"],
                attention_mask = prompt_enc["attention_mask"],
                max_new_tokens = self.data_args.max_new_tokens,
                do_sample      = False,
                eos_token_id   = self._tokenizer.eos_token_id,
            )

        prompt_len = prompt_enc["input_ids"].shape[1]
        preds = [
            self._tokenizer.decode(g[prompt_len:], skip_special_tokens=True).strip()
            for g in gen_ids
        ]

        # Encode preds và refs thành ids để trả về cho compute_metrics
        # Trainer truyền (predictions, label_ids) vào compute_metrics
        # Dùng object array để giữ string
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Numpy is required for returning predictions and references as arrays.")
        pred_arr = np.array(preds,    dtype=object)
        ref_arr  = np.array(answers,  dtype=object)

        return (loss, pred_arr, ref_arr)
    
    def log(self, logs: Any):
        """Override log to add custom wandb logging for auxiliary losses."""
        super().log(logs)
        
        # Log additional info to wandb if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb_logs = {}
            if "ce_loss" in logs:
                wandb_logs["ce_loss"] = logs["ce_loss"]
            if "lb_loss" in logs:
                wandb_logs["lb_loss"] = logs["lb_loss"]
            if "z_loss" in logs:
                wandb_logs["z_loss"] = logs["z_loss"]
            if "loss" in logs:
                wandb_logs["total_loss"] = logs["loss"]
            if wandb_logs:
                wandb.log(wandb_logs, step=self.state.global_step)


# ════════════════════════════════════════════════════════════════════════════════
# 8. compute_metrics callback
# ════════════════════════════════════════════════════════════════════════════════
def make_compute_metrics(data_args: DataArguments):
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        preds = list(eval_pred.predictions)
        refs  = list(eval_pred.label_ids)
        # Lọc None (prediction_loss_only steps)
        preds = [str(p) for p in preds if p is not None]
        refs  = [str(r) for r in refs  if r is not None]
        if not preds:
            return {}
        return compute_ngram_metrics(preds, refs, data_args.ngram_n)
    return compute_metrics


# ════════════════════════════════════════════════════════════════════════════════
# 9. Model setup
# ════════════════════════════════════════════════════════════════════════════════
def configuration4model(model_args: ModelArguments):
    pass

def setup_model(model_args: ModelArguments, data_args: DataArguments):
    """
    Setup model with MoE insertion and freeze configuration.
    
    Args:
        model_args: Model configuration (model_name, num_experts, freeze options, etc.)
        data_args: Data configuration (max_seq_length, etc.)
    
    Returns:
        model: Model with MoE inserted
        tokenizer: Tokenizer
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 0: Log configuration
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("MODEL CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model name: {model_args.model_name}")
    logger.info(f"MoE Config: num_experts={model_args.num_experts}, top_k={model_args.top_k}, num_shared={model_args.num_shared}")
    logger.info(f"Loss weights: w_lb={model_args.w_lb}, w_z={model_args.w_z}")
    logger.info(f"Hidden sizes: vis_hidden={model_args.vis_hidden}, lm_hidden={model_args.lm_hidden}")
    logger.info("=" * 80)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: Load tokenizer and model
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info(f"Loading tokenizer from {model_args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name, 
        trust_remote_code=True, 
        use_fast=False
    )
    tokenizer.model_max_length = data_args.max_seq_length
    logger.info(f"Tokenizer loaded. Max length set to {data_args.max_seq_length}")

    logger.info(f"Loading model from {model_args.model_name}...")
    model = AutoModel.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    logger.info("Model loaded successfully")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: Apply freeze options
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("APPLYING FREEZE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"freeze_vision: {model_args.freeze_vision}")
    logger.info(f"freeze_llm: {model_args.freeze_llm}")
    logger.info(f"freeze_mlp1: {model_args.freeze_mlp1}")
    
    # Freeze vision encoder (InternVIT)
    if model_args.freeze_vision:
        if hasattr(model, 'vision_model'):
            for p in model.vision_model.parameters():
                p.requires_grad = False
            logger.info("✓ Vision encoder (InternVIT) FROZEN")
        else:
            logger.warning("⚠ Vision model not found, skipping freeze")
    else:
        logger.info("✓ Vision encoder TRAINABLE")
    
    # Freeze language model
    if model_args.freeze_llm:
        if hasattr(model, 'language_model'):
            for p in model.language_model.parameters():
                p.requires_grad = False
            logger.info("✓ Language model (LLM) FROZEN")
        else:
            logger.warning("⚠ Language model not found, skipping freeze")
    else:
        logger.info("✓ Language model TRAINABLE")
    
    # Freeze mlp1 (if exists before MoE insertion)
    if model_args.freeze_mlp1:
        if hasattr(model, 'mlp1'):
            for p in model.mlp1.parameters():
                p.requires_grad = False
            logger.info("✓ Original mlp1 FROZEN")
        else:
            logger.warning("⚠ mlp1 not found, skipping freeze")
    else:
        logger.info("✓ Original mlp1 TRAINABLE")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: Insert MoE module
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("INSERTING MOE MODULE")
    logger.info("=" * 80)
    logger.info(f"Creating QueryConditionedMoE with:")
    logger.info(f"  - hidden_size (vis_hidden): {model_args.vis_hidden}")
    logger.info(f"  - query_size (lm_hidden): {model_args.lm_hidden}")
    logger.info(f"  - num_experts: {model_args.num_experts}")
    logger.info(f"  - top_k: {model_args.top_k}")
    logger.info(f"  - num_shared: {model_args.num_shared}")
    logger.info(f"  - dropout: {model_args.moe_dropout}")
    
    moe = QueryConditionedMoE(
        hidden_size = model_args.vis_hidden,
        query_size  = model_args.lm_hidden,
        num_experts = model_args.num_experts,
        top_k       = model_args.top_k,
        num_shared  = model_args.num_shared,
        dropout     = model_args.moe_dropout,
    ).to(dtype=torch.bfloat16)
    logger.info("✓ MoE module created")

    # Wrap mlp1 with MoE
    logger.info("Wrapping mlp1 with MoE...")
    original_mlp1 = model.mlp1
    model.mlp1 = MLP1WithMoE(original_mlp1, moe)
    logger.info("✓ mlp1 wrapped with MoE")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 4: Unfreeze MoE (always trainable, regardless of freeze_mlp1)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("UNFREEZING MOE MODULE (Always trainable)")
    logger.info("=" * 80)
    for p in model.mlp1.moe.parameters():
        p.requires_grad = True
    logger.info("✓ MoE module set to TRAINABLE")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5: Log trainable parameters
    # ═══════════════════════════════════════════════════════════════════════════
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable / total if total > 0 else 0
    
    logger.info("=" * 80)
    logger.info("PARAMETER COUNT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total parameters:     {total:>15,}")
    logger.info(f"Trainable parameters: {trainable:>15,} ({trainable_pct:>6.2f}%)")
    logger.info(f"Frozen parameters:    {total - trainable:>15,} ({100 - trainable_pct:>6.2f}%)")
    logger.info("=" * 80)
    print(f"\n📊 Trainable: {trainable:,} / {total:,}  ({trainable_pct:.2f}%)\n")

    return model, tokenizer


# ════════════════════════════════════════════════════════════════════════════════
# 10. Main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    # ═══════════════════════════════════════════════════════════════════════════
    # Parse all arguments
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("Parsing command-line arguments...")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logger.info("=" * 80)
    logger.info("PARSED ARGUMENTS")
    logger.info("=" * 80)
    logger.info(f"Model Arguments:")
    logger.info(f"  model_name: {model_args.model_name}")
    logger.info(f"  num_experts: {model_args.num_experts}")
    logger.info(f"  top_k: {model_args.top_k}")
    logger.info(f"  w_lb: {model_args.w_lb}, w_z: {model_args.w_z}")
    logger.info(f"Data Arguments:")
    logger.info(f"  meta_path: {data_args.meta_path}")
    logger.info(f"  conv_style: {data_args.conv_style}")
    logger.info(f"Training Arguments:")
    logger.info(f"  output_dir: {training_args.output_dir}")
    logger.info(f"  num_train_epochs: {training_args.num_train_epochs}")
    logger.info(f"  learning_rate: {training_args.learning_rate}")
    logger.info("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════════
    # Initialize Weights & Biases
    # ═══════════════════════════════════════════════════════════════════════════
    if WANDB_AVAILABLE:
        logger.info("Initializing Weights & Biases...")
        wandb.init(
            project="vintern-moe-training",
            name=f"moe_lr{training_args.learning_rate}_bs{training_args.per_device_train_batch_size}_expert{model_args.num_experts}",
            config={
                "model_name": model_args.model_name,
                "num_experts": model_args.num_experts,
                "top_k": model_args.top_k,
                "num_shared": model_args.num_shared,
                "w_lb": model_args.w_lb,
                "w_z": model_args.w_z,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_epochs": training_args.num_train_epochs,
                "freeze_vision": model_args.freeze_vision,
                "freeze_llm": model_args.freeze_llm,
                "freeze_mlp1": model_args.freeze_mlp1,
            },
            tags=["moe", "vintern", "vqa"],
        )
        logger.info("✓ Weights & Biases initialized")
    else:
        logger.warning("⚠ Weights & Biases not available. Install with: pip install wandb")

    # ═══════════════════════════════════════════════════════════════════════════
    # Setup model (with actual arguments)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nSetting up model with MoE...")
    model, tokenizer = setup_model(model_args, data_args)

    # ═══════════════════════════════════════════════════════════════════════════
    # Initialize TCSLoader for remote image loading (optional)
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        from petrel_client.client import Client
        from petrel_client.common.config import Config
        tcs_loader = TCSLoader('~/petreloss.conf')
        logger.info('✓ TCSLoader initialized for remote image loading')
    except ImportError:
        logger.info('⚠ petrel_client is not installed. Using PIL to load images.')
        tcs_loader = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Build training dataset from meta file
    # ═══════════════════════════════════════════════════════════════════════════
    if data_args.meta_path is None:
        raise ValueError("data_args.meta_path must be specified to load datasets via LazySupervisedDataset")
    
    logger.info(f"\nBuilding datasets from meta file: {data_args.meta_path}...")
    train_dataset = build_datasets(
        data_args=data_args,
        tokenizer=tokenizer,
        tcs_loader=tcs_loader,
        model=model,
        group_by_length=training_args.group_by_length if hasattr(training_args, 'group_by_length') else False,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
    )
    logger.info(f"✓ Training dataset created with {len(train_dataset)} samples")
    eval_dataset = None  # Eval dataset handled separately if needed

    # ═══════════════════════════════════════════════════════════════════════════
    # Create Trainer
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nCreating MoETrainer...")
    trainer = MoETrainer(
        model            = model,
        args             = training_args,
        data_collator    = concat_pad_data_collator,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        compute_metrics  = make_compute_metrics(data_args),
        model_args       = model_args,
        data_args        = data_args,
        tokenizer        = tokenizer,
    )

    # Add WandbCallback for detailed loss tracking
    if WANDB_AVAILABLE:
        trainer.add_callback(WandbCallback())
        logger.info("✓ WandbCallback added to trainer")

    # ═══════════════════════════════════════════════════════════════════════════
    # Start training
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Training for {training_args.num_train_epochs} epochs")
    logger.info(f"Total loss: ce_loss + {model_args.w_lb}*lb_loss + {model_args.w_z}*z_loss")
    logger.info("=" * 80 + "\n")
    
    trainer.train()

    # ═══════════════════════════════════════════════════════════════════════════
    # Save weights and cleanup
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    
    # Save MoE weights
    moe_save_path = os.path.join(training_args.output_dir, "moe_weights.pt")
    torch.save(model.mlp1.moe.state_dict(), moe_save_path)
    logger.info(f"✓ MoE weights saved to {moe_save_path}")

    # Finish wandb run
    if WANDB_AVAILABLE:
        wandb.finish()
        logger.info("✓ Weights & Biases run finished")
    
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()