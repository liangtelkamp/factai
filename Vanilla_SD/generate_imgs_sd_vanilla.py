# Packages to install in the iti-gen environment:
# conda install diffusers==0.11.1
# conda install transformers scipy ftfy accelerate
import argparse
import torch
from diffusers import StableDiffusionPipeline

def generate_images(prompt, output_dir, num_images, num_inference_steps, seed, model_name):
    """
    Generate images using Stable Diffusion Pipeline. 
    Code inherited form https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb
    """
    # Set seed for Python random module
    torch.manual_seed(seed)
    
    # Set seed for PyTorch on CUDA (GPU) if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)

    # Initialize generator
    generator = torch.Generator("cuda")

    # Generate and save images
    for i in range(num_images):
        image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
        image.save(f"{output_dir}/Baseline_person_{i:03d}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion Pipeline.")
    parser.add_argument("--prompt", type=str, default="a headshot of a person", help="Prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="/imgs", help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=200, help="Number of images to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for image generation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="Name of thye Stable Diffusion model")

    args = parser.parse_args()
    
    generate_images(args.prompt, args.output_dir, args.num_images, args.num_inference_steps, args.seed, args.model_name)
