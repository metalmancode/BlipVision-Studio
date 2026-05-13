import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. ENHANCEMENT: Hardware Acceleration
# Automatically use GPU (CUDA) if available, Apple Silicon (MPS) if on Mac, or fallback to CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model on {device}...")

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image: Image.Image, text_prompt: str) -> str:
    """
    Generates a caption for the provided image.
    """
    try:
        if image is None:
            return "Please upload an image first."
            
        # 2. ENHANCEMENT: Conditional Captioning
        # BLIP can take an optional prompt to guide the caption generation
        if text_prompt.strip():
            inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        else:
            inputs = processor(images=image, return_tensors="pt").to(device)
            
        # 3. ENHANCEMENT: Better generation parameters
        # max_new_tokens ensures the generated caption isn't cut off early
        outputs = model.generate(**inputs, max_new_tokens=50)
        
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"An error occurred: {str(e)}"

# 4. ENHANCEMENT: Modern UI with gr.Blocks
# gr.Blocks provides a more flexible and customizable layout than gr.Interface
with gr.Blocks(title="Enhanced Image Captioning") as app:
    gr.Markdown("# 🖼️ AI Image Captioning with BLIP")
    gr.Markdown("Upload an image to automatically generate a descriptive caption using the Salesforce BLIP model.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image")
            # Optional prompt to guide the model
            text_prompt = gr.Textbox(
                label="Optional Starting Prompt", 
                placeholder="e.g., 'A picture of a...'"
            )
            submit_btn = gr.Button("Generate Caption", variant="primary")
            
        with gr.Column(scale=1):
            output_caption = gr.Textbox(label="Generated Caption", lines=6)
            
    # Link the UI elements to the python function
    submit_btn.click(
        fn=generate_caption,
        inputs=[input_image, text_prompt],
        outputs=output_caption
    )

if __name__ == "__main__":
    # Launch the application
    app.launch(server_name="127.0.0.1", server_port=7860)
