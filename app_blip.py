import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. ENHANCEMENT: Hardware Acceleration
# Automatically use GPU (CUDA) if available, Apple Silicon (MPS) if on Mac, or fallback to CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model on {device}...")

# Load the processor and model (Upgraded to LARGE for better details)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def analyze_image(image: Image.Image, text_prompt: str):
    """
    Generates a detailed caption and SEO keywords for the provided image.
    """
    try:
        if image is None:
            return "Please upload an image first.", ""
            
        # --- 1. DETAILED CAPTION / QA ---
        if text_prompt.strip():
            # If the user provides a prompt/question, we treat it as a prefix for the AI to complete
            inputs_caption = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
            outputs_caption = model.generate(
                **inputs_caption, 
                max_new_tokens=150, 
                num_beams=5,
                repetition_penalty=1.5,
                temperature=0.9,
                do_sample=True
            )
            raw_caption = processor.decode(outputs_caption[0], skip_special_tokens=True)
            
            # CRITICAL FIX: Remove the prompt from the beginning of the result
            if raw_caption.lower().startswith(text_prompt.lower()):
                detailed_caption = raw_caption[len(text_prompt):].strip()
                if not detailed_caption:
                    detailed_caption = raw_caption
            else:
                detailed_caption = raw_caption
        else:
            # Unconditional detailed captioning
            inputs_caption = processor(images=image, return_tensors="pt").to(device)
            outputs_caption = model.generate(
                **inputs_caption, 
                max_new_tokens=150, 
                min_length=40, 
                num_beams=5,
                repetition_penalty=1.5
            )
            detailed_caption = processor.decode(outputs_caption[0], skip_special_tokens=True)
        
        # --- 2. SEO KEYWORDS ---
        # A more descriptive prompt for professional tags
        kw_prompt = "keywords for a professional social media post about this image:"
        inputs_kw = processor(images=image, text=kw_prompt, return_tensors="pt").to(device)
        outputs_kw = model.generate(
            **inputs_kw, 
            max_new_tokens=60,
            do_sample=True,
            top_k=50,
            temperature=0.9,
            repetition_penalty=1.2
        )
        keywords_raw = processor.decode(outputs_kw[0], skip_special_tokens=True)
        
        # Clean up prompt prefix
        keywords_raw = keywords_raw.replace(kw_prompt, "").strip()
        
        # --- 3. FORMAT AS HASHTAGS ---
        # Stopwords to filter out for better SEO
        STOPWORDS = {"tags", "are", "not", "tool", "to", "describe", "a", "the", "in", "on", "at", "with", "is", "for", "of", "and", "or", "as", "this", "that", "it", "my", "your", "be"}
        
        import re
        # Split by commas, spaces, and punctuation
        words = re.split(r'[,\s\.\!\?\:\;]+', keywords_raw)
        
        # Filter: lowercase, length > 2, and not a stopword
        clean_words = []
        seen = set()
        for w in words:
            w_clean = w.lower().strip()
            if len(w_clean) > 2 and w_clean not in STOPWORDS and w_clean not in seen:
                clean_words.append(f"#{w_clean}")
                seen.add(w_clean)
        
        hashtags = " ".join(clean_words)
        
        # Fallback if it failed to generate good tags
        if not hashtags:
            hashtags = "#data #business #technology #workspace #office"
        
        return detailed_caption, hashtags
        
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

# 4. ENHANCEMENT: Modern UI with gr.Blocks
with gr.Blocks(title="BlipVision Studio") as app:
    gr.Markdown("# 🖼️ BlipVision Studio: Pro Image Analyzer")
    gr.Markdown("Upload an image to get high-detail descriptions and optimized SEO hashtags.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image")
            text_prompt = gr.Textbox(
                label="Custom Focus (Optional)", 
                placeholder="e.g., 'describe the monitors' or 'focus on the colors'"
            )
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column(scale=1):
            output_caption = gr.Textbox(
                label="Detailed Explanation", 
                lines=5, 
                buttons=["copy"]
            )
            output_keywords = gr.Textbox(
                label="SEO Hashtags", 
                lines=3, 
                buttons=["copy"]
            )
            
    # Link the UI elements
    submit_btn.click(
        fn=analyze_image,
        inputs=[input_image, text_prompt],
        outputs=[output_caption, output_keywords]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
