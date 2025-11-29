import torch, json
import argparse
from diffusers import AutoPipelineForText2Image

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run diffusion model with CT prompts")
parser.add_argument("--prompt_path", type=str, default="./prompt_generator/prompt/",
                    help="Path to the prompt JSON folder")
parser.add_argument("--prompt_filename", type=str, default="ct_prompts_2wise.json",
                    help="Path to the prompt JSON file")
parser.add_argument("--model", type=str, default="sdxl",
                    choices=["sdxl", "sdxl-turbo", "sdxl-lightning", "pixart-sigma"],
                    help="Model to use for image generation")
parser.add_argument("--batch_size", type=int, default=10,
                    help="Number of images to generate per prompt")
parser.add_argument("--start_from", type=int, default=0,
                    help="prompt index to start from")
args = parser.parse_args()

PROMPT_PATH = args.prompt_path + args.prompt_filename
APPEND_PROMPT = "exposed face, looking at the camera, ultra quality, sharp focus, 8k, photorealistic, intricate details, hdr, high resolution"
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
	MODEL_NAME, torch_dtype=torch.float16 # Deprecated warning is expected, passing dtype does NOT work
).to("cuda")
pipeline.unet.to(memory_format=torch.channels_last)

# IF WANTED, ADD torch.Generator HERE FOR REPRODUCIBILITY

prompts = []
prompt_file = json.load(open(PROMPT_PATH, "r"))
print(f"Processing {len(prompt_file['test_cases'])} prompts from {PROMPT_PATH}:{args.start_from}")

with torch.inference_mode():
    for case in prompt_file['test_cases'][args.start_from:]:
        count=0
        for _ in range( BATCH_SIZE // 2 ): 
            images = pipeline(
                prompt=f"{case['prompt']}, {APPEND_PROMPT}",
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=15, # FAIntBench
                guidance_scale=7,
                height=1024,
                width=1024,
                num_images_per_prompt=2, # 3 이상으로 안됨. 
                safety_checker=None,
            ).images
            for img in images:
                img.save(f"./outputs/{case['id']}_{count}.png")
                count += 1
        print(f"Completed prompt #{case['id']}: '{case['prompt']}' with {count} images.")