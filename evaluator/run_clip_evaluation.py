import argparse
import json
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


PROMPT_DIR = "../prompt_generator/prompt" 
OUTPUT_ROOT_DIR = "../outputs" 
RESULT_DIR = "clip_results"

MODEL_LIST = {
    "openai-base": "openai/clip-vit-base-patch32",
    "openai-large": "openai/clip-vit-large-patch14",
    "laion-l-14": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
}

MODEL = "openai-large" 

def evaluate_image(model, processor, image_path, full_prompt, attribute_prompts):
    try:
        image = Image.open(image_path).convert("RGB")
        all_prompts = [full_prompt] + attribute_prompts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        similarity_scores = outputs.logits_per_image[0]
        return {
            "full_prompt_score": similarity_scores[0].item(),
            "attribute_scores": {
                attr_prompt: score.item() for attr_prompt, score in zip(attribute_prompts, similarity_scores[1:])
            }
        }
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error evaluating image '{image_path}': {e}")
        return None


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Runs n-wise intersectional evaluation.")
    parser.add_argument("-n", type=int, required=True, help="Number for n-wise analysis (e.g., 2 or 3)")
    args = parser.parse_args()
    n = args.n_wise
    print(f"--- Starting {n}-wise Intersectional Evaluation ---")

    # Path Generation
    prompt_filename = f"ct_prompts_{n}wise.json"
    image_subfolder = f"outputs_{n}wise"
    prompt_filepath = os.path.join(PROMPT_DIR, prompt_filename)
    image_dir = os.path.join(OUTPUT_ROOT_DIR, image_subfolder)
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_filepath = os.path.join(RESULT_DIR, f"scores_{n}wise.json")
    
    if not os.path.exists(prompt_filepath):
        print(f"Error: Prompt file ({prompt_filepath}) not found.")
        return
    if not os.path.exists(image_dir):
        print(f"Error: Image directory ({image_dir}) not found.")
        return

    # Model Loading
    print("\nLoading CLIP model...")
    try:
        if MODEL not in MODEL_LIST:
            print(f"Error: Selected model '{MODEL}' is not defined in MODEL_LIST.")
            return
        
        current_model_id = MODEL_LIST[MODEL]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(current_model_id).to(device)
        processor = CLIPProcessor.from_pretrained(current_model_id)
        print(f"Using CLIP model: {current_model_id} on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prompt Data Loading
    with open(prompt_filepath, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    test_cases = {case['id']: case for case in prompt_data['test_cases']}

    # Evaluation
    all_results = []
    print(f"\nEvaluating images in '{image_dir}' folder...")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        try:
            case_id = int(filename.split('_')[0])
            case_info = test_cases.get(case_id)
            if not case_info:
                print(f"Warning: Prompt info for ID {case_id} (from {filename}) not found.")
                continue

            # Generate attribute prompts
            attribute_prompts = []
            factors_metadata = {factor['name']: factor['labels'] for factor in prompt_data.get("factors", [])}
            for factor_name, factor_value in case_info['assignment'].items():
                if factor_name in factors_metadata:
                    attribute_prompts.append(f"a photo of a {factor_value}")
                    attribute_prompts.append(factor_value)
            attribute_prompts = list(set([p for p in attribute_prompts if p.strip()]))
            
            # Evaluate the image
            full_prompt = case_info['prompt']
            image_path = os.path.join(image_dir, filename)
            scores = evaluate_image(model, processor, image_path, full_prompt, attribute_prompts)
            
            if scores:
                result_object = {
                    "image_id": case_id,
                    "image_filename": filename,
                    "full_prompt": full_prompt,
                    "full_prompt_score": scores['full_prompt_score'],
                    "assigned_attributes": case_info['assignment'],
                    "attribute_scores": scores['attribute_scores']
                }
                all_results.append(result_object)

        except (ValueError, IndexError):
            print(f"Warning: Could not extract ID from filename '{filename}'")
        except Exception as e:
            print(f"Unexpected error processing file '{filename}': {e}")

    # Save Results
    if not all_results:
        print("\nNo images evaluated or all evaluations failed.")
        return
    
    all_results.sort(key=lambda x: (x['image_id'], x['image_filename']))

    print(f"\nSaving evaluation results to {result_filepath}...")
    try:
        with open(result_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()