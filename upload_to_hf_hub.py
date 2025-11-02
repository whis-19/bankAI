import os
from pathlib import Path
from huggingface_hub import HfApi, HfFolder, ModelCard, logging
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoModelForImageClassification,
    AutoTokenizer, 
    AutoConfig,
    AutoImageProcessor
)
import argparse
from typing import Optional, List, Dict, Any
import shutil
from tqdm import tqdm
import torch
import json

# Set up logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class ModelUploader:
    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the ModelUploader with optional Hugging Face token."""
        self.api = HfApi()
        if hf_token:
            self.api.set_access_token(hf_token)
        self.hf_token = hf_token or HfFolder.get_token()
        
        if not self.hf_token:
            raise ValueError(
                "No Hugging Face token provided. Please set the HUGGING_FACE_HUB_TOKEN environment variable "
                "or pass the token as an argument."
            )

    def find_model_dirs(self, base_dir: str = ".") -> List[str]:
        """Search for model directories in the project."""
        model_dirs = []
        
        # Common model directory patterns
        common_patterns = [
            "**/OUTPUT_*",
            "**/models",
            "**/checkpoints",
            "**/saved_models",
            "**/final_model"
        ]
        
        base_path = Path(base_dir).resolve()
        
        for pattern in common_patterns:
            for path in base_path.glob(pattern):
                if path.is_dir() and any((path / f).exists() for f in ["pytorch_model.bin", "model.safetensors", "config.json"]):
                    model_dirs.append(str(path))
                    
        # Also check for specific output directories from the training scripts
        specific_paths = [
            base_path / "OUTPUT_Recipe" / "final_model",
            # Add other specific paths as needed
        ]
        
        for path in specific_paths:
            if path.exists() and path.is_dir():
                model_dirs.append(str(path))
        
        return list(dict.fromkeys(model_dirs))  # Remove duplicates while preserving order

    def get_model_info(self, model_dir: str) -> Dict[str, Any]:
        """Get model information based on the model directory."""
        model_type = None
        base_model = None
        
        # Check for model type based on directory name or contents
        if 'Recipe' in model_dir:
            model_type = 'recipe'
            base_model = 'gpt2'
        elif 'T5' in model_dir or 'Summary' in model_dir:
            model_type = 'summary'
            base_model = 't5-base'  # or whatever base model was used
        elif 'ViT' in model_dir or 'Food41' in model_dir:
            model_type = 'image'
            base_model = 'google/vit-base-patch16-224'  # or whatever base model was used
            
        return {
            'type': model_type,
            'base_model': base_model,
            'path': model_dir
        }

    def create_model_card(self, model_name: str, model_dir: str) -> str:
        """Create a model card with information specific to the model type."""
        model_info = self.get_model_info(model_dir)
        model_type = model_info['type']
        base_model = model_info['base_model']
        
        if model_type == 'recipe':
            return f"""---
license: mit
language:
- en
tags:
- recipe-generation
- gpt2
---

# {model_name}

This is a fine-tuned version of the GPT-2 model for recipe generation.

## Model Details

- **Model type:** GPT-2
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from:** {base_model}

## Intended Uses & Limitations

This model is intended for generating cooking recipes based on input prompts. It was fine-tuned on a recipe dataset.

### How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Generate a recipe
input_prompt = "Chocolate Chip Cookies Recipe:"
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=300, num_return_sequences=1)
generated_recipe = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_recipe)
```
"""
        elif model_type == 'summary':
            return f"""---
license: mit
language:
- en
tags:
- text-summarization
- t5
---

# {model_name}

This is a fine-tuned version of the T5 model for text summarization.

## Model Details

- **Model type:** T5
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from:** {base_model}

## Intended Uses & Limitations

This model is intended for summarizing text content. It was fine-tuned on a custom summarization dataset.

### How to Use

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

def summarize(text, max_length=142):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
summary = summarize("Your long text here...")
print(summary)
```
"""
        elif model_type == 'image':
            return f"""---
license: mit
tags:
- image-classification
- food
- vit
---

# {model_name}

This is a fine-tuned Vision Transformer (ViT) model for food image classification.

## Model Details

- **Model type:** Vision Transformer (ViT)
- **License:** MIT
- **Finetuned from:** {base_model}
- **Dataset:** Food-101 (subset)

## Intended Uses & Limitations

This model is intended for classifying food images into various food categories.

### How to Use

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained("{model_name}")
model = AutoModelForImageClassification.from_pretrained("{model_name}")

# Load and preprocess the image
image = Image.open("path_to_your_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
# Get the predicted class
print(f"Predicted class: {{model.config.id2label[predicted_class_idx]}}")
```
"""
        else:
            # Default model card for unknown model types
            return f"""---
license: mit
---

# {model_name}

This is a fine-tuned model.

## Model Details

- **Finetuned from:** {base_model or 'Unknown'}

## Intended Uses & Limitations

This model is intended for specific tasks it was fine-tuned for.
"""

    def upload_model(
        self,
        model_dir: str,
        repo_name: str,
        organization: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload model",
    ) -> str:
        """Upload a model to the Hugging Face Hub."""
        # Create repository if it doesn't exist
        repo_id = f"{organization}/{repo_name}" if organization else repo_name
        
        try:
            # Check if the model directory contains necessary files
            required_files = ["config.json"]
            model_files = ["pytorch_model.bin", "model.safetensors"]
            
            if not any((Path(model_dir) / f).exists() for f in model_files):
                raise ValueError(f"No model weights found in {model_dir}. Expected one of {model_files}")
                
            # Create a temporary directory to store the model card
            temp_dir = Path("temp_model_upload")
            temp_dir.mkdir(exist_ok=True, parents=True)
            
            try:
                # Copy model files to temp directory
                for item in os.listdir(model_dir):
                    src = Path(model_dir) / item
                    if src.is_file():
                        shutil.copy2(src, temp_dir / item)
                
                # Create and save model card
                model_card_content = self.create_model_card(repo_name, model_dir)
                with open(temp_dir / "README.md", "w", encoding="utf-8") as f:
                    f.write(model_card_content)
                
                # Upload to Hub
                self.api.create_repo(
                    repo_id=repo_id,
                    private=private,
                    exist_ok=True,
                    repo_type="model"
                )
                
                # Upload all files
                for root, _, files in os.walk(temp_dir):
                    for file in tqdm(files, desc="Uploading files"):
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(temp_dir)
                        self.api.upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=str(relative_path),
                            repo_id=repo_id,
                            commit_message=commit_message,
                        )
                
                return f"https://huggingface.co/{repo_id}"
                
            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            raise Exception(f"Failed to upload model: {str(e)}")

def get_model_paths_from_app() -> Dict[str, str]:
    """Extract model paths from app.py"""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
    model_paths = {}
    
    try:
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract MODEL_PATHS dictionary from app.py
        start = content.find('MODEL_PATHS = {')
        if start != -1:
            end = content.find('}', start) + 1
            model_paths_str = content[start:end]
            # Use eval to safely parse the dictionary (since we control the input)
            model_paths = eval(model_paths_str.split('=', 1)[1].strip())
    except Exception as e:
        print(f"Warning: Could not extract model paths from app.py: {e}")
    
    return model_paths

def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned models to Hugging Face Hub")
    parser.add_argument(
        "--repo-prefix",
        type=str,
        default="food-ai",
        help="Prefix for repository names on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Organization name on the Hugging Face Hub (optional).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether the models should be private on the Hub.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token. If not provided, will use the token from ~/.huggingface/token.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to a specific model directory to upload (overrides auto-detection).",
    )
    
    args = parser.parse_args()
    
    try:
        uploader = ModelUploader(hf_token=args.hf_token)
        
        if args.model_dir:
            # Upload specific model
            if not os.path.exists(args.model_dir):
                raise ValueError(f"Model directory not found: {args.model_dir}")
            model_paths = {'custom': args.model_dir}
        else:
            # Get model paths from app.py or use default paths
            model_paths = get_model_paths_from_app()
            
            # Fallback to default paths if not found in app.py
            if not model_paths:
                # Find ViT model directory (any directory ending with ViT)
                vit_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.lower().endswith('vit')]
                vit_path = os.path.join(vit_dirs[0], 'final_model') if vit_dirs else None
                
                model_paths = {
                    'recipe': './OUTPUT_Recipe/final_model',
                    'summary': './OUTPUT_T5_Summary/final_model',
                    'image': vit_path or './OUTPUT_Images_ViT/final_model' or './OUTPUT_Food41_ViT/final_model'  # Use found path or default
                }
        
        # Filter out non-existent paths
        model_paths = {name: path for name, path in model_paths.items() 
                      if os.path.exists(path) and os.path.isdir(path)}
        
        if not model_paths:
            print("❌ No valid model directories found. Please check the paths in app.py or specify with --model-dir.")
            return 1
        
        print(f"Found {len(model_paths)} model directories:")
        for name, path in model_paths.items():
            print(f"- {name}: {path}")
        
        # Upload each model
        for model_name, model_dir in model_paths.items():
            repo_name = f"{args.repo_prefix}-{model_name}"
            
            print(f"\n{'='*50}")
            print(f"Uploading {model_name} model from {model_dir}...")
            
            try:
                model_url = uploader.upload_model(
                    model_dir=model_dir,
                    repo_name=repo_name,
                    organization=args.organization,
                    private=args.private,
                    commit_message=f"Upload {repo_name}",
                )
                
                print(f"\n✅ Successfully uploaded {model_name} model to: {model_url}")
                print(f"Model directory: {model_dir}")
                
            except Exception as e:
                print(f"❌ Failed to upload {model_name} model: {str(e)}")
                continue
        
        print("\n" + "="*50)
        print("✅ All models processed!")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
