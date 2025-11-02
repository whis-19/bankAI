# bankAI - AI-Powered Culinary Assistant

![bankAI Banner](https://via.placeholder.com/1200x400/0a0a2a/ffffff?text=bankAI+Culinary+Assistant)

An intelligent application that combines multiple AI models to enhance your culinary experience. From generating recipes to identifying food items in images, bankAI is your one-stop solution for all things food.

## 🍽️ Features

### 1. Recipe Generation
- Generate creative recipes from simple ingredients
- Customize generation with temperature, top-k, and top-p parameters
- Beautifully formatted recipe output with clear sections

### 2. Text Summarization
- Condense long recipes or cooking articles into key points
- Adjustable summary length and detail level
- Perfect for quick recipe reviews

### 3. Food Image Classification
- Identify food items from images
- Get confidence scores for predictions
- Fun and interactive results display

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bankai-culinary.git
   cd bankai-culinary
   ```

2. Create and activate a virtual environment:
   ```bash
   conda create -n bankai python=3.8
   conda activate bankai
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

## 🛠️ Configuration

### Environment Variables
Create a `.env` file in the root directory with the following variables:
```
# Optional: Set to use CPU only
# FORCE_CPU=true

# Optional: HuggingFace Hub token (if using private models)
# HUGGINGFACE_HUB_TOKEN=your_token_here
```

## 📊 Models Used

| Feature | Model | Source |
|---------|-------|--------|
| Recipe Generation | GPT-2 Fine-tuned | [HuggingFace Hub](https://huggingface.co/whis-22/bankai-recipe) |
| Text Summarization | T5 Fine-tuned | [HuggingFace Hub](https://huggingface.co/whis-22/bankai-summary) |
| Food Classification | ViT Fine-tuned | [HuggingFace Hub](https://huggingface.co/whis-22/bankai-image) |

## 📱 Usage Examples

### Generating a Recipe
1. Select "Recipe Generation" from the sidebar
2. Enter your ingredients (e.g., "chicken, rice, vegetables")
3. Adjust generation parameters if needed
4. Click "Generate Recipe"

### Classifying Food from Image
1. Select "Food Classification" from the sidebar
2. Upload an image of food
3. Click "Classify Image"
4. View the top predictions with confidence scores

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for the Transformers library
- [Streamlit](https://streamlit.io/) for the amazing web framework
- The open-source community for their contributions

---

<div align="center">
  Made with ❤️ by Your Name | 2023
</div>
