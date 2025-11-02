import os
import json
from datetime import datetime
import pandas as pd
import ast  # For safely evaluating string-lists
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None
try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:
    rouge_scorer = None
try:
    import nltk  # type: ignore
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
except Exception:
    nltk = None
    sentence_bleu = None
    SmoothingFunction = None

# Optional KaggleHub imports
try:
    import kagglehub  # type: ignore
    from kagglehub.adapters import KaggleDatasetAdapter  # type: ignore
except Exception:  # pragma: no cover
    kagglehub = None
    KaggleDatasetAdapter = None

class Cfg:
    output_dir: str = './OUTPUT_Recipe'
    dataset_dir: str = './dataset'
    file_name: str = '3A2M_EXTENDED.csv'
    sample_fraction: float = 0.1
    random_state: int = 42
    val_size: float = 0.2
    num_epochs: int = 5
    per_device_train_bs: int = 8
    per_device_eval_bs: int = 8
    grad_accum: int = 8
    num_workers: int = min(16, (os.cpu_count() or 8))
    pin_memory: bool = True
    persistent_workers: bool = True
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = 'cosine'
    eval_sample_size: int = 50

cfg = Cfg()

def ensure_dirs(cfg: Cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'logs'), exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)

def load_or_download_df(cfg: Cfg) -> pd.DataFrame:
    dataset_dir = cfg.dataset_dir
    local_csv_path = os.path.join('.', cfg.file_name)
    dataset_csv_path = os.path.join(dataset_dir, cfg.file_name)
    if os.path.exists(dataset_csv_path):
        print(f"Loading dataset from {dataset_csv_path}...")
        return pd.read_csv(dataset_csv_path)
    if os.path.exists(local_csv_path):
        print(f"Loading dataset from {local_csv_path}...")
        return pd.read_csv(local_csv_path)
    if kagglehub is None or KaggleDatasetAdapter is None:
        raise RuntimeError("kagglehub is not installed. Install kagglehub or place the CSV under ./dataset")
    print("Local CSV not found. Loading from KaggleHub...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "nazmussakibrupol/3a2mext",
        cfg.file_name,
    )
    try:
        df.to_csv(local_csv_path, index=False, encoding='utf-8')
        df.to_csv(dataset_csv_path, index=False, encoding='utf-8')
        print(f"Saved dataset to {local_csv_path} and {dataset_csv_path}")
    except Exception as e:
        print(f"Warning: failed to save CSV copies: {e}")
    return df

def preprocess_df(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    df_clean = df[['title', 'NER', 'directions']].dropna().reset_index(drop=True)
    SAMPLE_FRACTION = cfg.sample_fraction
    _frac = max(0.0, min(float(SAMPLE_FRACTION), 1.0))
    df_sample = df_clean.sample(frac=_frac, random_state=cfg.random_state) if _frac < 1.0 else df_clean
    def format_recipe(row):
        try:
            title = row['title']
            ingredients_list = ast.literal_eval(row['NER'])
            ingredients_str = ", ".join(ingredients_list)
            directions_list = ast.literal_eval(row['directions'])
            directions_str = "\n".join(directions_list)
            if not title or not ingredients_str or not directions_str:
                return None
            return f"TITLE: {title}\nINGREDIENTS: {ingredients_str}\nRECIPE:\n{directions_str}"
        except (ValueError, SyntaxError):
            return None
    df_sample['text'] = df_sample.apply(format_recipe, axis=1)
    df_processed = df_sample.dropna(subset=['text']).reset_index(drop=True)
    return df_processed

class RecipeDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = {
            "input_ids": [],
            "attention_mask": []
        }
        print(f"Tokenizing {len(texts)} texts...")
        iterator = tqdm(texts, desc="Tokenizing", total=len(texts)) if tqdm else texts
        for text in iterator:
            tokenized = tokenizer(
                text + tokenizer.eos_token,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            self.encodings["input_ids"].append(torch.tensor(tokenized['input_ids']))
            self.encodings["attention_mask"].append(torch.tensor(tokenized['attention_mask']))
        print("Tokenization complete.")

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
        }
        item['labels'] = item['input_ids'].clone()
        return item

if __name__ == "__main__":

    # Load the full dataset
    print("Loading full dataset...")
    ensure_dirs(cfg)
    try:
        df = load_or_download_df(cfg)
    except Exception:
        print("Failed to load dataset. Please check the path and try again.")

    print(f"Loaded {len(df)} total recipes.")

    # Preprocess
    df_processed = preprocess_df(df, cfg)
    print(f"Successfully processed {len(df_processed)} recipes.")
    if len(df_processed) > 0:
        print("\n--- Example of Formatted Text ---")
        print(df_processed['text'][0])

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Split
    train_texts, val_texts = train_test_split(
        df_processed['text'], test_size=cfg.val_size, random_state=cfg.random_state
    )

    # Datasets
    train_dataset = RecipeDataset(train_texts.tolist(), tokenizer)
    val_dataset = RecipeDataset(val_texts.tolist(), tokenizer)
    print(f"\nCreated {len(train_dataset)} training examples.")
    print(f"Created {len(val_dataset)} validation examples.")

    # Device / precision
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Using device: {device.upper()} ---")
    use_fp16 = device == 'cuda'

    # Model
    print("Loading pre-trained GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Training args
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.output_dir, 'checkpoints'),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_bs,
        per_device_eval_batch_size=cfg.per_device_eval_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        eval_strategy="steps",
        logging_steps=100,
        save_steps=500,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        load_best_model_at_end=True,
        fp16=use_fp16,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting the fine-tuning process...")
    trainer.train()

    # Save
    print("Training complete. Saving the final model...")
    final_model_path = os.path.join(cfg.output_dir, 'final_model')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model and tokenizer saved to {final_model_path}")

    # Example generation using the saved model
    input_prompt = "Vegetarian Pasta with Tomatoes and Basil"
    recipe_output = generate_recipe(final_model_path, input_prompt, max_length=300)
    if recipe_output:
        print("\n--- GENERATED RECIPE ---")
        print(recipe_output)

    # Automatic evaluation (ROUGE-L, BLEU) on a sample of the validation set
    try:
        val_df = df_processed.loc[val_texts.index][['title', 'text']]
        evaluate_model(
            model,
            tokenizer,
            val_df,
            device,
            sample_size=cfg.eval_sample_size,
            output_dir=cfg.output_dir,
        )
    except Exception as e:
        print(f"Evaluation skipped due to error: {e}")

def generate_recipe(model_path, prompt, max_length=300):
    """
    Loads a fine-tuned GPT-2 model and tokenizer from a specified path
    and generates text based on a given prompt.
    """
    
    print(f"Loading model and tokenizer from {model_path}...")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer and model
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the path is correct and all files (config.json, model.safetensors, etc.) are present.")
        return

    # Set the pad token to the eos token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Move model to the selected device
    model.to(device)
    model.eval()  # Set model to evaluation mode (important!)

    print("Model loaded. Encoding prompt...")
    
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    print("Generating recipe...")

    # Generate text
    with torch.no_grad():  # Disable gradient calculations for inference
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.7,       # Makes the output less random
            top_k=50,              # Considers the top 50 most likely words
            top_p=0.95,            # Uses nucleus sampling
            do_sample=True,        # Enables sampling
            num_return_sequences=1,# We just want one recipe
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated sequence
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text

def generate_with_model(model, tokenizer, prompt, device, max_length=300):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, val_df, device, sample_size=50, output_dir=None):
    if len(val_df) == 0:
        print("No validation data to evaluate.")
        return {}
    n = min(sample_size, len(val_df))
    eval_sample = val_df.sample(n=n, random_state=0) if len(val_df) > sample_size else val_df
    rouges = []
    bleus = []
    use_rouge = rouge_scorer is not None
    use_bleu = sentence_bleu is not None
    if not use_rouge and not use_bleu:
        print("Skipping automatic metrics (install rouge-score and nltk for ROUGE/BLEU).")
        return {}
    scorer = rouge_scorer.Scorer(['rougeLsum'], use_stemmer=True) if use_rouge else None
    smooth = SmoothingFunction().method1 if use_bleu else None
    rows_iter = tqdm(list(eval_sample.iterrows()), desc="Evaluating", total=len(eval_sample)) if tqdm else eval_sample.iterrows()
    for _, row in rows_iter:
        prompt = row['title']
        reference = row['text']
        hypothesis = generate_with_model(model, tokenizer, prompt, device, max_length=300)
        if use_rouge:
            r = scorer.score(reference, hypothesis)['rougeLsum'].fmeasure
            rouges.append(r)
        if use_bleu:
            ref_tokens = reference.split()
            hyp_tokens = hypothesis.split()
            try:
                b = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
            except Exception:
                b = 0.0
            bleus.append(b)
    metrics = {}
    if rouges:
        metrics['rougeL_f1_avg'] = sum(rouges)/len(rouges)
        print(f"ROUGE-L (F1) avg on {n} samples: {metrics['rougeL_f1_avg']:.4f}")
    if bleus:
        metrics['bleu_avg'] = sum(bleus)/len(bleus)
        print(f"BLEU avg on {n} samples: {metrics['bleu_avg']:.4f}")
    metrics['num_samples'] = n
    if output_dir:
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(output_dir, 'logs', f'metrics_{ts}.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"Saved metrics to {path}")
        except Exception as e:
            print(f"Failed to save metrics JSON: {e}")
    return metrics