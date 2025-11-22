import torch, json
import argparse
from diffusers import AutoPipelineForText2Image

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run diffusion model with CT prompts")
parser.add_argument("--prompt_path", type=str, default="./prompt_generator/prompt/ct_prompts_2wise.json",
                    help="Path to the prompt JSON file")
parser.add_argument("--model", type=str, default="sdxl-turbo",
                    choices=["sdxl", "sdxl-turbo", "sdxl-lightning", "pixart-sigma"],
                    help="Model to use for image generation")
parser.add_argument("--batch_size", type=int, default=10,
                    help="Number of images to generate per prompt")
args = parser.parse_args()

PROMPT_PATH = args.prompt_path
APPEND_PROMPT = "exposed face, looking at the camera, ultra quality, sharp focus, 8k, photorealistic, intricate details, hdr, high resolution, professional lighting"
NEGATIVE_PROMPT = "lowres, bad anatomy, error body, error hands, missing fingers, cropped, worst quality, low quality, jpeg artifacts"
MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sdxl-lightning": "ByteDance/SDXL-Lightning", # use 1, 2, 4, or 8 step
    "pixart-sigma": "PixArt-alpha/PixArt-Sigma" # may not be available in Diffusers?
}
MODEL_NAME = MODELS[args.model]
BATCH_SIZE = args.batch_size

# Boring settings part
torch.backends.cuda.matmul.allow_tf32 = True

pipeline = AutoPipelineForText2Image.from_pretrained(
	MODEL_NAME, dtype=torch.float16
).to("cuda")
pipeline.unet.to(memory_format=torch.channels_last)

# IF WANTED, ADD torch.Generator HERE FOR REPRODUCIBILITY

prompts = []
prompt_file = json.load(open(PROMPT_PATH, "r"))
print(f"Processing {len(prompt_file)} prompts from {PROMPT_PATH}")

with torch.inference_mode():
    for case in prompt_file["test_cases"]:
        images = pipeline(
            prompt=f"{case['prompt']}, {APPEND_PROMPT}",
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=15, # FAIntBench
            guidance_scale=7,
            height=1024,
            width=1024,
            num_images_per_prompt=BATCH_SIZE,
        ).images
        for i, img in enumerate(images):
            img.save(f"./outputs/{case['id']}_{i}.png")
        print(f"Completed prompt: {case['prompt']}")
print("All prompts processed.")