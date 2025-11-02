import os
import streamlit as st
import torch
from tqdm.auto import tqdm
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    ViTForImageClassification,
    AutoImageProcessor,
    BitsAndBytesConfig
)
from PIL import Image
import numpy as np
from functools import partial

# Configure tqdm to work with Streamlit
tqdm = partial(tqdm, position=0, leave=True)

# Model paths (Hugging Face Hub)
MODEL_PATHS = {
    'recipe': 'whis-22/bankai-recipe',
    'summary': 'whis-22/bankai-summary',
    'image': 'whis-22/bankai-image'
}

# Initialize session state for models
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_progress_callback(progress_bar, total_size):
    def update_progress(bytes_downloaded):
        progress = min(int(bytes_downloaded / total_size * 100), 100)
        progress_bar.progress(progress, text=f"Downloading model: {progress}%")
    return update_progress

@st.cache_resource(show_spinner=False)
def load_model(model_type):
    """Load the specified model with caching and progress tracking"""
    model_path = MODEL_PATHS[model_type]
    progress_bar = st.progress(0, text="Initializing model loading...")
    
    try:
        # Configure model loading with progress tracking
        config = {
            'pretrained_model_name_or_path': model_path,
            'device_map': 'auto',
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
            'low_cpu_mem_usage': True,
        }
        
        if model_type == 'recipe':
            progress_bar.progress(10, text="Loading recipe model...")
            model = GPT2LMHeadModel.from_pretrained(**config)
            progress_bar.progress(70, text="Loading tokenizer...")
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            progress_bar.progress(100, text="Model loaded successfully!")
            return {'model': model, 'tokenizer': tokenizer}
        
        elif model_type == 'summary':
            progress_bar.progress(10, text="Loading summarization model...")
            model = T5ForConditionalGeneration.from_pretrained(**config)
            progress_bar.progress(70, text="Loading tokenizer...")
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            progress_bar.progress(100, text="Model loaded successfully!")
            return {'model': model, 'tokenizer': tokenizer}
        
        elif model_type == 'image':
            progress_bar.progress(10, text="Loading image classification model...")
            model = ViTForImageClassification.from_pretrained(**config)
            progress_bar.progress(70, text="Loading processor...")
            processor = AutoImageProcessor.from_pretrained(model_path)
            progress_bar.progress(100, text="Model loaded successfully!")
            return {'model': model, 'processor': processor}
            
    except Exception as e:
        progress_bar.error(f"Error loading model: {str(e)}")
        raise
    finally:
        # Give some time to read the final message before clearing
        import time
        time.sleep(1)
        progress_bar.empty()

def generate_recipe(prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.95):
    """Generate recipe using the fine-tuned GPT-2 model"""
    model_info = st.session_state.models['recipe']
    model = model_info['model']
    tokenizer = model_info['tokenizer']
    
    inputs = tokenizer(prompt, return_tensors='pt').to(st.session_state.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_summary(text, max_length=142, num_beams=4, length_penalty=2.0):
    """Generate summary using the fine-tuned T5 model"""
    model_info = st.session_state.models['summary']
    model = model_info['model']
    tokenizer = model_info['tokenizer']
    
    inputs = tokenizer(f"summarize: {text}", return_tensors='pt', truncation=True, max_length=1024).to(st.session_state.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def classify_image(image):
    """Classify food image using the fine-tuned ViT model"""
    model_info = st.session_state.models['image']
    model = model_info['model']
    processor = model_info['processor']
    
    inputs = processor(images=image, return_tensors='pt').to(st.session_state.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=3)  # Get top 3 predictions
    
    # Food-101 class names
    food101_classes = [
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
        'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
        'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
        'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
        'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
        'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
        'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
        'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
        'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
        'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
        'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
        'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
        'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
        'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
        'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
        'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
        'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
        'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
        'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
        'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
        'waffles'
    ]
    
    # Convert indices to class names and format output
    results = []
    for i in range(top_probs.shape[1]):
        class_idx = top_indices[0][i].item()
        class_name = food101_classes[class_idx].replace('_', ' ').title()
        prob = top_probs[0][i].item() * 100
        results.append((class_name, prob))
    
    return results

# Bleach-inspired Professional CSS
st.markdown("""
<style>
    .main {
        background-color: #0a0a2a;
        color: #e0e0e0;
    }
    
    .stApp {
        background-color: #0a0a2a;
    }
    
    .soul-title {
        color: #2a7fff;
        font-weight: 700;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    
    .soul-subtitle {
        color: #ff6b35;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 2em;
        font-weight: 300;
    }
    
    .stSidebar {
        background-color: #1a1a3a;
    }
    
    .model-section {
        background-color: #1a1a3a;
        border-left: 4px solid #2a7fff;
        padding: 2em;
        border-radius: 8px;
        margin: 1em 0;
    }
    
    .stButton>button {
        background-color: #2a7fff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.7em 1.5em;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1a5fcc;
    }
    
    .stSelectbox>div>div {
        background-color: #1a1a3a;
        border: 1px solid #2a7fff;
        color: white;
    }
    
    .stTextArea>div>div>textarea {
        background-color: #1a1a3a;
        border: 1px solid #ff6b35;
        color: white;
    }
    
    .stSlider>div>div>div {
        background-color: #2a7fff;
    }
    
    .stFileUploader>div>div {
        background-color: #1a1a3a;
        border: 2px dashed #ff6b35;
        color: white;
    }
    
    .prediction-card {
        background-color: #252545;
        border: 1px solid #2a7fff;
        border-radius: 5px;
        padding: 1em;
        margin: 0.5em 0;
    }
    
    .footer {
        text-align: center;
        padding: 1em;
        background-color: #1a1a3a;
        color: #2a7fff;
        margin-top: 2em;
        border-top: 1px solid #2a7fff;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(page_title="bankAI - Model Platform", layout="wide")

# Header
st.markdown('<div class="soul-title">bankAI</div>', unsafe_allow_html=True)
st.markdown('<div class="soul-subtitle">Advanced AI Model Platform</div>', unsafe_allow_html=True)

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Recipe Generation", "Text Summarization", "Food Classification"]
)

# Load selected model
if model_type == "Recipe Generation":
    if 'recipe' not in st.session_state.models:
        with st.spinner('Loading Recipe Generation Model...'):
            st.session_state.models['recipe'] = load_model('recipe')
    
    st.markdown('<div class="model-section">', unsafe_allow_html=True)
    st.header("Recipe Generation")
    prompt = st.text_area("Enter ingredients or dish name:", "Chicken and rice")
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max Length", 50, 500, 200, 50)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    with col2:
        top_k = st.slider("Top-k", 1, 100, 50, 5)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    
    if st.button("Generate Recipe"):
        with st.spinner("Cooking up your recipe... 🍳"):
            recipe = generate_recipe(prompt, max_length, temperature, top_k, top_p)
            st.markdown("---")
            st.markdown("## 🍽️ Generated Recipe")
            st.markdown("---")
            # Split the recipe into lines and format each line
            formatted_recipe = ""
            for line in recipe.split('\n'):
                line = line.strip()
                if line.startswith('Title:') or line.startswith('Title :'):
                    formatted_recipe += f"### {line.replace('Title:', '').replace('Title :', '').strip()}\n\n"
                elif line.startswith('Ingredients:'):
                    formatted_recipe += "### 🛒 Ingredients\n"
                elif line.startswith('Instructions:'):
                    formatted_recipe += "\n### 📝 Instructions\n"
                elif line.startswith('-'):
                    formatted_recipe += f"- {line[1:].strip()}\n"
                elif line and any(char.isalpha() for char in line):
                    # If it's a numbered step
                    if line[0].isdigit() and (line[1] == '.' or line[1] == ')'):
                        formatted_recipe += f"\n**{line}**\n"
                    else:
                        formatted_recipe += f"{line}  \n"  # Double space at end for markdown line break
            
            st.markdown(formatted_recipe, unsafe_allow_html=True)
            st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

elif model_type == "Text Summarization":
    if 'summary' not in st.session_state.models:
        with st.spinner('Loading Text Summarization Model...'):
            st.session_state.models['summary'] = load_model('summary')
    
    st.markdown('<div class="model-section">', unsafe_allow_html=True)
    st.header("Text Summarization")
    text = st.text_area("Enter text to summarize:", """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
    the natural intelligence displayed by animals including humans. AI research has been 
    defined as the field of study of intelligent agents, which refers to any system that 
    perceives its environment and takes actions that maximize its chance of achieving its goals.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max Length", 50, 200, 142, 10)
        num_beams = st.slider("Number of Beams", 1, 8, 4, 1)
    with col2:
        length_penalty = st.slider("Length Penalty", 0.1, 3.0, 2.0, 0.1)
    
    if st.button("Generate Summary"):
        with st.spinner("Summarizing text... ✍️"):
            summary = generate_summary(text, max_length, num_beams, length_penalty)
            st.markdown("---")
            st.markdown("## 📝 Summary")
            st.markdown("---")
            st.markdown(f"{summary}")
            st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

elif model_type == "Food Classification":
    if 'image' not in st.session_state.models:
        with st.spinner('Loading Food Classification Model...'):
            st.session_state.models['image'] = load_model('image')
    
    st.markdown('<div class="model-section">', unsafe_allow_html=True)
    st.header("Food Classification")
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Classify Image'):
            with st.spinner('Analyzing food... 🍽️'):
                predictions = classify_image(image)
                st.markdown("---")
                st.markdown("## 🔍 Classification Results")
                st.markdown("---")
                
                # Create columns for better layout
                col1, col2 = st.columns([1, 3])
                
                # Display top prediction with emoji and confidence
                top_food, top_prob = predictions[0]
                with col1:
                    st.metric(label="Most Likely", 
                            value=top_food, 
                            delta=f"{top_prob:.1f}% confidence")
                
                # Display all predictions in a nice table
                with col2:
                    st.markdown("### All Predictions")
                    for i, (food, prob) in enumerate(predictions, 1):
                        # Create a progress bar for visual effect
                        progress = int(prob)
                        html_content = (
                            f"<div style='margin-bottom: 10px;'>"
                            f"<strong>{i}. {food}</strong>"
                            f"<div style='background: #e0e0e0; border-radius: 5px; height: 20px; margin: 5px 0;'>"
                            f"<div style='background: #4CAF50; width: {progress}%; height: 100%; border-radius: 5px;'>"
                            f"<span style='color: white; padding-left: 10px; line-height: 20px;'>{prob:.1f}%</span>"
                            "</div></div></div>"
                        )
                        st.markdown(html_content, unsafe_allow_html=True)
                
                # Add some fun food-related emojis based on confidence
                if top_prob > 80:
                    st.balloons()
                    st.markdown("🎉 Wow! I'm very confident about this one! 🎉")
                elif top_prob > 60:
                    st.snow()
                    st.markdown("🤔 I'm pretty sure about this, but not 100% certain.")
                else:
                    st.markdown("🤷 I'm not entirely sure, but here's my best guess!")
                    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About bankAI
bankAI provides access to specialized AI models:

- **Recipe Generation**: Create recipes from ingredients
- **Text Summarization**: Condense long texts
- **Food Classification**: Identify food items from images

Select a model to begin.
""")

# Footer
st.markdown(
    """
    <div class="footer">
    <p>bankAI Platform | Advanced Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)