# BlipVision Studio 🖼️

An advanced image analysis and SEO optimization tool powered by the Salesforce **BLIP-Large** model and **Gradio**.

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [How it Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Privacy & Security](#-privacy--security)
- [Technical Stack](#-technical-stack)

## 🌟 Overview
BlipVision Studio is a professional-grade tool designed for content creators, SEO specialists, and developers. It uses state-of-the-art vision models to extract detailed descriptions and generate high-value SEO hashtags from any image, all while running 100% locally on your machine.

## ✨ Key Features
- **Pro Image Analysis**: Utilizes the **BLIP-Large** model for high-fidelity, descriptive captions.
- **Smart SEO Hashtags**: Automatically generates optimized, professional hashtags tailored for business and technology contexts.
- **Theme Booster**: Intelligently identifies image themes (e.g., Fintech, Data, Workspace) and appends industry-standard tags.
- **Stopword Filtering**: Automatically cleans hashtags to remove filler words, ensuring only high-value keywords remain.
- **One-Click Copy**: Built-in copy buttons for easy integration into your workflow.
- **Hardware Acceleration**: Full support for NVIDIA GPUs (CUDA) and Apple Silicon (MPS).

## 🛠️ How it Works
1. **Upload**: Drag and drop an image into the studio.
2. **Focus (Optional)**: Provide a specific question or starting phrase to guide the AI.
3. **Analyze**: The AI processes the image using your local hardware.
4. **Copy**: Grab your detailed explanation and SEO-optimized hashtags instantly.

## 🚀 Installation
### Prerequisites
- Python 3.9 or higher
- (Recommended) A dedicated GPU for faster processing

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/metalmancode/BlipVision-Studio.git
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage
Run the application locally:
```bash
python app_blip.py
```
Open your browser and navigate to: **http://127.0.0.1:7860**

## 🔒 Privacy & Security
- **Local-First**: All image processing is performed locally. No data ever leaves your machine.
- **No Cloud Uploads**: Your images are never stored or used for model training.
- **Offline Ready**: Works without an internet connection once the model is downloaded.

## 💻 Technical Stack
- **AI Model**: [Salesforce BLIP-Large](https://huggingface.co/Salesforce/blip-image-captioning-large)
- **Framework**: [Gradio](https://gradio.app/)
- **Libraries**: Hugging Face Transformers, PyTorch, PIL.

## 📜 License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---
Developed by [metalmancode](https://github.com/metalmancode)
