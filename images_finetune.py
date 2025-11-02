import os
import json
from datetime import datetime
from typing import List, Tuple
import inspect

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import functional as TF
from PIL import Image

from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
)

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

try:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    plt = None
    np = None

try:
    import kagglehub  # type: ignore
    try:
        # Newer style
        from kagglehub import KaggleDatasetAdapter  # type: ignore
    except Exception:
        # Older style
        from kagglehub.adapters import KaggleDatasetAdapter  # type: ignore
except Exception:
    kagglehub = None
    KaggleDatasetAdapter = None


class Cfg:
    output_dir: str = './OUTPUT_Food41_ViT'
    dataset_root: str = './dataset/food41/images'
    sample_fraction: float = 1.0
    random_state: int = 42
    val_size: float = 0.2
    num_epochs: int = 3
    per_device_train_bs: int = 8
    per_device_eval_bs: int = 8
    grad_accum: int = 4
    num_workers: int = min(16, (os.cpu_count() or 8))
    pin_memory: bool = True
    persistent_workers: bool = True
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = 'cosine'
    model_name: str = 'google/vit-base-patch16-224-in21k'
    eval_sample_size: int = 100


cfg = Cfg()


def ensure_dirs(cfg: Cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'logs'), exist_ok=True)
    # dataset dirs
    os.makedirs(os.path.dirname(os.path.abspath(cfg.dataset_root)), exist_ok=True)
    os.makedirs(cfg.dataset_root, exist_ok=True)


def try_download_food41_with_kagglehub(target_dir: str) -> str:
    """Download Food-41 via KaggleHub. If possible, place under target_dir; otherwise return cache path."""
    if kagglehub is None or KaggleDatasetAdapter is None:
        return ''
    try:
        # Trigger download/sync to KaggleHub cache
        cache_path = kagglehub.load_dataset(
            KaggleDatasetAdapter.FILES,  # type: ignore[attr-defined]
            'kmader/food41',
            '',
        )
        if not (isinstance(cache_path, str) and os.path.isdir(cache_path)):
            return ''

        # Try to move into target_dir if target is empty/non-existent
        if not os.path.exists(target_dir):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(target_dir)), exist_ok=True)
                os.rename(cache_path, target_dir)
                return target_dir
            except Exception:
                pass

        # If target exists and is empty, move contents
        try:
            os.makedirs(target_dir, exist_ok=True)
            if len(os.listdir(target_dir)) == 0:
                # Move top-level contents
                for name in os.listdir(cache_path):
                    src = os.path.join(cache_path, name)
                    dst = os.path.join(target_dir, name)
                    try:
                        os.rename(src, dst)
                    except Exception:
                        # If rename fails (cross-device), fall back to leave in cache
                        return cache_path
                return target_dir
        except Exception:
            pass

        # Fall back to using cache path directly
        return cache_path
    except Exception:
        return ''


def find_image_root(base_dir: str) -> str:
    candidates = [
        base_dir,
        os.path.join(base_dir, 'images'),
        os.path.join(base_dir, 'Food-101', 'images'),
        os.path.join(base_dir, 'food41', 'images'),
    ]
    for p in candidates:
        if os.path.isdir(p):
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subdirs) > 1:
                return p
    return ''


class ViTImageDataset(Dataset):
    def __init__(self, image_folder: datasets.ImageFolder, processor: AutoImageProcessor, train: bool = True):
        self.ds = image_folder
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, label = self.ds.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.train:
                if torch.randint(0, 2, (1,)).item():
                    img = TF.hflip(img)
            enc = self.processor(images=img, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(label)
        return item


def stratified_split_indices(targets: List[int], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    import random
    rnd = random.Random(seed)
    by_class = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(idx)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for y, idxs in by_class.items():
        rnd.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_fraction)) if len(idxs) > 0 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def stratified_sample_indices(targets: List[int], frac: float, seed: int) -> List[int]:
    import random
    rnd = random.Random(seed)
    by_class = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(idx)
    sampled: List[int] = []
    for y, idxs in by_class.items():
        n = len(idxs)
        k = n if frac >= 1.0 else max(1, int(n * max(0.0, min(frac, 1.0))))
        rnd.shuffle(idxs)
        sampled.extend(idxs[:k])
    return sampled


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


def save_confusion_matrix(y_true, y_pred, class_names: List[str], out_dir: str):
    try:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'confusion_matrix.json'), 'w', encoding='utf-8') as f:
            json.dump({'labels': class_names, 'matrix': cm.tolist()}, f, ensure_ascii=False, indent=2)
        if plt is not None and np is not None:
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)), xticklabels=class_names, yticklabels=class_names, ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
            plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=200)
            plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    ensure_dirs(cfg)

    base_root = cfg.dataset_root
    if not os.path.isdir(base_root):
        alt = try_download_food41_with_kagglehub(base_root)
        if alt:
            base_root = alt

    image_root = find_image_root(base_root)
    if not image_root:
        print("Food-41 images not found. Please set cfg.dataset_root to the directory containing class subfolders.")
        raise SystemExit(1)

    processor = AutoImageProcessor.from_pretrained(cfg.model_name)
    base_folder = datasets.ImageFolder(root=image_root)
    class_names = base_folder.classes
    print(f"Detected {len(class_names)} classes under {image_root}.")

    all_targets = [y for _, y in base_folder.samples]
    sampled_indices = stratified_sample_indices(all_targets, cfg.sample_fraction, cfg.random_state)
    sampled_subset = Subset(base_folder, sampled_indices)
    sampled_targets = [all_targets[i] for i in sampled_indices]
    train_idx, val_idx = stratified_split_indices(sampled_targets, cfg.val_size, cfg.random_state)
    train_subset = Subset(sampled_subset, train_idx)
    val_subset = Subset(sampled_subset, val_idx)

    train_ds = ViTImageDataset(image_folder=train_subset.dataset, processor=processor, train=True) if isinstance(train_subset.dataset, datasets.ImageFolder) else ViTImageDataset(image_folder=base_folder, processor=processor, train=True)
    val_ds = ViTImageDataset(image_folder=val_subset.dataset, processor=processor, train=False) if isinstance(val_subset.dataset, datasets.ImageFolder) else ViTImageDataset(image_folder=base_folder, processor=processor, train=False)

    def subset_collate(subset: Subset):
        class _Wrap(Dataset):
            def __init__(self, subset: Subset, base: ViTImageDataset):
                self.subset = subset
                self.base = base

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, i):
                base_idx = self.subset.indices[i]
                return self.base[base_idx]

        return _Wrap(subset, ViTImageDataset(base_folder, processor, train=True))

    train_wrapped = subset_collate(train_subset)
    val_wrapped = subset_collate(val_subset)

    id2label = {i: c for i, c in enumerate(class_names)}
    label2id = {c: i for i, c in id2label.items()}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")
    use_fp16 = device == 'cuda'

    model = ViTForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
    )

    # Build TrainingArguments with compatibility for different transformers versions
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    def supports(name: str) -> bool:
        return name in supported

    ta_kwargs = {
        'output_dir': os.path.join(cfg.output_dir, 'checkpoints'),
        'num_train_epochs': cfg.num_epochs,
        'per_device_train_batch_size': cfg.per_device_train_bs,
        'per_device_eval_batch_size': cfg.per_device_eval_bs,
        'gradient_accumulation_steps': cfg.grad_accum,
        'logging_steps': 100,
        'save_steps': 500,
    }
    if supports('evaluation_strategy'):
        ta_kwargs['evaluation_strategy'] = 'steps'
    elif supports('eval_strategy'):
        ta_kwargs['eval_strategy'] = 'steps'
    if supports('warmup_ratio'):
        ta_kwargs['warmup_ratio'] = cfg.warmup_ratio
    if supports('lr_scheduler_type'):
        ta_kwargs['lr_scheduler_type'] = cfg.lr_scheduler_type
    if supports('load_best_model_at_end'):
        ta_kwargs['load_best_model_at_end'] = True
    if supports('fp16'):
        ta_kwargs['fp16'] = use_fp16
    if supports('report_to'):
        ta_kwargs['report_to'] = 'none'

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_wrapped,
        eval_dataset=val_wrapped,
        tokenizer=processor,
        compute_metrics=compute_metrics_fn,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    print("Saving model and processor...")
    final_model_path = os.path.join(cfg.output_dir, 'final_model')
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)

    print("Predicting on validation set for detailed metrics...")
    preds_output = trainer.predict(val_wrapped)
    y_true = preds_output.label_ids.tolist() if hasattr(preds_output, 'label_ids') else []
    y_pred = preds_output.predictions.argmax(axis=-1).tolist() if hasattr(preds_output, 'predictions') else []
    if y_true and y_pred:
        acc = accuracy_score(y_true, y_pred)
        print(f"Validation accuracy: {acc:.4f}")
        save_confusion_matrix(y_true, y_pred, class_names, os.path.join(cfg.output_dir, 'plots'))
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(cfg.output_dir, 'logs', f'metrics_{ts}.json'), 'w', encoding='utf-8') as f:
                json.dump({'accuracy': acc}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    print("Sample predictions:")
    n_show = min(10, len(val_subset))
    for i in range(n_show):
        base_idx = val_subset.indices[i]
        path, true_lbl = base_folder.samples[base_idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            inputs = processor(images=img, return_tensors='pt')
        with torch.no_grad():
            logits = model(**{k: v.to(model.device) for k, v in inputs.items()}).logits
            pred = int(logits.argmax(dim=-1).cpu().item())
        print(f"{os.path.basename(path)} | true={class_names[true_lbl]} | pred={class_names[pred]}")

