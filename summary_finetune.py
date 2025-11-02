import os
import json
from datetime import datetime
import inspect
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:
    rouge_scorer = None


class Cfg:
    output_dir: str = './OUTPUT_T5_Summary'
    dataset_dir: str = './dataset/cnn_dailymail'
    train_file: str = 'train.csv'
    val_file: str = 'validation.csv'
    test_file: str = 'test.csv'
    sample_fraction: float = 0.2  # Load only 20% of the data
    random_state: int = 42
    num_epochs: int = 3
    per_device_train_bs: int = 4
    per_device_eval_bs: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = 'cosine'
    model_name: str = 't5-small'
    max_source_len: int = 1024
    max_target_len: int = 128
    val_max_target_len: int = 142
    generation_max_len: int = 142
    eval_sample_size: int = 100


cfg = Cfg()


def ensure_dirs(cfg: Cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'logs'), exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing file: {path}')
    return pd.read_csv(path)


def load_dfs(cfg: Cfg):
    train_path = os.path.join(cfg.dataset_dir, cfg.train_file)
    val_path = os.path.join(cfg.dataset_dir, cfg.val_file)
    test_path = os.path.join(cfg.dataset_dir, cfg.test_file)
    train_df = _read_csv(train_path)
    val_df = _read_csv(val_path)
    test_df = _read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame(columns=['id', 'article', 'highlights'])
    required_cols = {'id', 'article', 'highlights'}
    for name, df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"{name} CSV must have columns: id, article, highlights")
    return train_df, val_df, test_df


def maybe_sample_df(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    _frac = max(0.0, min(float(frac), 1.0))
    return df.sample(frac=_frac, random_state=seed) if _frac < 1.0 else df


class SummarizationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: T5Tokenizer, max_source_len: int, max_target_len: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_src = max_source_len
        self.max_tgt = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = f"summarize: {str(row['article'])}"
        tgt_text = str(row['highlights'])
        src = self.tok(
            src_text,
            max_length=self.max_src,
            truncation=True,
            padding=False,
        )
        tgt = self.tok(
            tgt_text,
            max_length=self.max_tgt,
            truncation=True,
            padding=False,
        )
        item = {
            'input_ids': torch.tensor(src['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(src['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(tgt['input_ids'], dtype=torch.long),
        }
        return item


def compute_rouge_metrics(trainer: Seq2SeqTrainer, dataset: Dataset, sample_size: int = 100):
    if rouge_scorer is None or len(dataset) == 0:
        return {}
    n = min(sample_size, len(dataset))
    idxs = list(range(n))
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouges = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    iterator = tqdm(idxs, desc='Evaluating (ROUGE)', total=n) if tqdm else idxs
    model = trainer.model
    tok = trainer.tokenizer
    device = model.device
    
    for i in iterator:
        row = dataset.df.iloc[i]
        src_text = f"summarize: {str(row['article'])}"
        inputs = tok(src_text, return_tensors='pt', max_length=cfg.max_source_len, truncation=True).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, 
                max_length=cfg.generation_max_len,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        pred = tok.decode(gen_ids[0], skip_special_tokens=True)
        ref = str(row['highlights'])
        s = scorer.score(ref, pred)
        for k in rouges.keys():
            rouges[k].append(s[k].fmeasure)
    return {f'{k}_f1_avg': float(sum(v) / len(v)) for k, v in rouges.items()}


if __name__ == '__main__':
    ensure_dirs(cfg)

    # Load and sample data - using 20% as per sample_fraction
    train_df, val_df, test_df = load_dfs(cfg)
    train_df = maybe_sample_df(train_df, cfg.sample_fraction, cfg.random_state)
    val_df = maybe_sample_df(val_df, cfg.sample_fraction, cfg.random_state)
    
    print(f"Sample sizes - Train: {len(train_df)}, Val: {len(val_df)}")

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = T5Tokenizer.from_pretrained(cfg.model_name)
    model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)

    train_ds = SummarizationDataset(train_df, tokenizer, cfg.max_source_len, cfg.max_target_len)
    val_ds = SummarizationDataset(val_df, tokenizer, cfg.max_source_len, cfg.val_max_target_len)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16 = device == 'cuda'

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
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
        'predict_with_generate': True,
        'generation_max_length': cfg.generation_max_len,
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

    training_args = Seq2SeqTrainingArguments(**ta_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print('Starting training...')
    trainer.train()

    print('Evaluating...')
    try:
        eval_metrics = trainer.evaluate(max_length=cfg.generation_max_len)
    except TypeError:
        eval_metrics = trainer.evaluate()
    rouge_metrics = compute_rouge_metrics(trainer, val_ds, sample_size=cfg.eval_sample_size)
    metrics = {**eval_metrics, **rouge_metrics}
    print(metrics)

    print('Saving model and tokenizer...')
    final_model_path = os.path.join(cfg.output_dir, 'final_model')
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(cfg.output_dir, 'logs', f'metrics_{ts}.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    def generate_summary(model_path: str, article: str, max_length: int = 142):
        tok = T5Tokenizer.from_pretrained(model_path)
        mdl = T5ForConditionalGeneration.from_pretrained(model_path)
        inputs = tok(f'summarize: {article}', return_tensors='pt', truncation=True, max_length=cfg.max_source_len)
        with torch.no_grad():
            ids = mdl.generate(**inputs, max_length=max_length)
        return tok.decode(ids[0], skip_special_tokens=True)

    if len(val_df) > 0:
        ex = val_df.iloc[0]['article']
        print('\n--- SAMPLE GENERATION ---')
        print(generate_summary(final_model_path, ex)[:1000])
