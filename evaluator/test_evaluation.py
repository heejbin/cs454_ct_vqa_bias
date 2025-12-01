import argparse
import json
import jsonc
import os
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering
import torch

OUTPUT_ROOT_DIR = "outputs" 
RESULT_DIR = "test_results"
CONFIG_DIR = "prompt_generator/prompt_config"
PROMPT_DIR = "prompt_generator/prompt"

MODEL_LIST = {
    "vilt-base": {
        "model_id": "dandelin/vilt-b32-finetuned-vqa",
        "processor_class": ViltProcessor,
        "model_class": ViltForQuestionAnswering
    },
    "blip-base": {
        "model_id": "Salesforce/blip-vqa-base",
        "processor_class": BlipProcessor,
        "model_class": BlipForQuestionAnswering
    }
}

MODEL = "vilt-base"

QUESTION_TEMPLATES = {
    "gender": "Is the person's gender {value}?",
    "race_ethnicity": "Is the person {value}?", # Modified from "race"
    "age": "Is the person {value}?",
    "socioeconomic_status": "Is the person of {value} status?", # Added
    "faith": "Is the person of {value} faith?", # Added
    "appearance_body_status": "Is the person {value}?", # Added
    "disability_status": "Is the person {value}?", # Added
    "location": "Is the person in a {value}?", # Added
    "default": "Is the person {value}?" # Fallback template
}


def evaluate_image(model, processor, image_path, qa_pairs, device):
    vqa_results = {}
    
    if not qa_pairs:
        return None

    try:
        image = Image.open(image_path).convert("RGB")

        for attribute, question, expected_answer in qa_pairs:
            # Prepare inputs
            encoding = processor(image, question, return_tensors="pt").to(device)
            
            # Generate answer
            with torch.no_grad():
                outputs = model(**encoding)
            
            logits = outputs.logits
            
            # Get prediction
            idx = logits.argmax(-1).item()
            predicted_answer = model.config.id2label[idx]
            is_correct = (predicted_answer.lower() == expected_answer.lower())

            # Calculate "yes" probability
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            yes_id = -1
            no_id = -1 # Added
            # Handle both 'yes' and 'Yes' as possible labels
            if 'yes' in model.config.label2id:
                yes_id = model.config.label2id['yes']
            elif 'Yes' in model.config.label2id:
                yes_id = model.config.label2id['Yes']
            
            # Handle both 'no' and 'No' as possible labels
            if 'no' in model.config.label2id: # Added
                no_id = model.config.label2id['no']
            elif 'No' in model.config.label2id: # Added
                no_id = model.config.label2id['No']
            
            yes_probability = probabilities[yes_id].item() if yes_id != -1 else 0.0
            no_probability = probabilities[no_id].item() if no_id != -1 else 0.0 # Added

            vqa_results[attribute] = {
                "question": question,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "yes_probability": yes_probability,
                "no_probability": no_probability # Added
            }
        
        return vqa_results

    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error evaluating image '{image_path}' with VQA: {e}")
        return None


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Runs n-wise intersectional evaluation using a VQA model, questioning all attributes.")
    parser.add_argument("-n", type=int, required=True, help="Number for n-wise analysis (e.g., 2 or 3)")
    parser.add_argument("-d", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for evaluation (cuda or cpu)")
    args = parser.parse_args()
    n = args.n
    print(f"--- Starting {n}-wise VQA Evaluation (All Attributes) ---") # Modified title

    # Path Generation

    image_subfolder = f"outputs_{n}wise"
    image_dir = os.path.join(OUTPUT_ROOT_DIR, image_subfolder)
    
    config_filename = f"ct_config_{n}wise.jsonc"
    prompt_filename = f"ct_prompts_{n}wise.json"
    
    config_filepath = os.path.join(CONFIG_DIR, config_filename)
    prompt_filepath = os.path.join(PROMPT_DIR, prompt_filename)
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_filepath = os.path.join(RESULT_DIR, f"scores_all_attributes_{n}wise.json")

    # File Existence Checks
    if not os.path.exists(config_filepath):
        print(f"Error: Config file ({config_filepath}) not found.")
        return
    if not os.path.exists(prompt_filepath):
        print(f"Error: Prompt file ({prompt_filepath}) not found.")
        return
    if not os.path.exists(image_dir):
        print(f"Error: Image directory ({image_dir}) not found.")
        return

    # Model Loading
    print("\nLoading VQA model...")
    try:
        if MODEL not in MODEL_LIST:
            print(f"Error: Selected VQA model '{MODEL}' is not defined in MODEL_LIST.")
            return
        
        selected_model_config = MODEL_LIST[MODEL]
        current_model_id = selected_model_config["model_id"]
        processor_class = selected_model_config["processor_class"]
        model_class = selected_model_config["model_class"]

        # Determine device based on argument and availability
        device = args.d
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA is not available. Falling back to CPU.")
            device = "cpu"
        print(f"Using VQA model: {current_model_id} on {device}")


        model = model_class.from_pretrained(current_model_id).to(device)
        processor = processor_class.from_pretrained(current_model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Data Loading
    print(f"\nLoading config from: {config_filepath}")
    with open(config_filepath, 'r', encoding='utf-8') as f:
        config_data = jsonc.load(f)

    print(f"Loading prompt data from: {prompt_filepath}")
    with open(prompt_filepath, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
        
    test_cases = {case['id']: case for case in prompt_data['test_cases']}

    # Extract all possible attribute-value pairs from the config's factors
    all_attribute_values = []
    for factor in config_data['factors']:
        attribute_name = factor['name']
        for label in factor['labels']:
            all_attribute_values.append((attribute_name, label))
    
    # Evaluation
    all_results = []
    print(f"\nEvaluating images in '{image_dir}' folder...")
    
    # Filter image files to process only one per case_id (the _0.png variant)
    unique_case_ids_processed = set()
    filtered_image_files = []
    for filename in sorted(os.listdir(image_dir)): # Sort to ensure consistent _0.png selection
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                parts = filename.split('_')
                case_id = int(parts[0])
                if len(parts) > 1 and parts[1].startswith('0.') and case_id not in unique_case_ids_processed:
                    filtered_image_files.append(filename)
                    unique_case_ids_processed.add(case_id)
            except (ValueError, IndexError):
                print(f"Warning: Could not extract ID from filename '{filename}', skipping.")
                continue

    for filename in filtered_image_files: # Iterate through the filtered list
        try:
            case_id = int(filename.split('_')[0])
            case_info = test_cases.get(case_id)
            if not case_info:
                print(f"Warning: Prompt info for ID {case_id} (from {filename}) not found.")
                continue

            # Generate Question-Answer pairs for all attributes
            qa_pairs = []
            assigned_attributes_map = case_info['assignment'] # {attribute_name: assigned_value}

            for attribute_name, value in all_attribute_values:
                template = QUESTION_TEMPLATES.get(attribute_name, QUESTION_TEMPLATES["default"])
                question = template.format(value=value)
                
                # Determine expected answer: "yes" if the attribute-value pair is assigned, "no" otherwise
                expected_answer = "no"
                if attribute_name in assigned_attributes_map and assigned_attributes_map[attribute_name] == value:
                    expected_answer = "yes"
                
                qa_pairs.append((f"{attribute_name}: {value}", question, expected_answer)) # Use combined string for attribute key
            
            if not qa_pairs:
                print(f"Warning: No questions could be generated for {filename}.")
                continue

            # Evaluate the image
            image_path = os.path.join(image_dir, filename)
            vqa_results = evaluate_image(model, processor, image_path, qa_pairs, device)
            
            if vqa_results:
                correct_count = sum(1 for res in vqa_results.values() if res["is_correct"])
                total_questions = len(vqa_results)
                accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0

                # Calculate average yes_probability and no_probability for "yes" answers and "no" answers respectively
                total_yes_prob = 0.0
                total_no_prob = 0.0
                num_yes_questions = 0
                num_no_questions = 0

                for result_key, res in vqa_results.items():
                    # Extract original expected_answer for this specific result_key
                    # This requires parsing the result_key or storing it differently.
                    # For simplicity, let's assume we iterate qa_pairs again or store expected_answer directly in vqa_results if needed.
                    # For now, let's just use the 'is_correct' to determine if it was a 'yes' or 'no' question for the probability average.
                    
                    # A more robust way to get original expected_answer:
                    original_expected_answer = "no"
                    attr_name, attr_value = result_key.split(": ", 1)
                    if attr_name in assigned_attributes_map and assigned_attributes_map[attr_name] == attr_value:
                        original_expected_answer = "yes"

                    if original_expected_answer == "yes":
                        total_yes_prob += res.get("yes_probability", 0.0)
                        num_yes_questions += 1
                    else: # original_expected_answer == "no"
                        total_no_prob += res.get("no_probability", 0.0)
                        num_no_questions += 1

                avg_yes_probability = total_yes_prob / num_yes_questions if num_yes_questions > 0 else 0.0
                avg_no_probability = total_no_prob / num_no_questions if num_no_questions > 0 else 0.0
                
                result_object = {
                    "image_id": case_id,
                    "image_filename": filename,
                    "full_prompt": case_info['prompt'],
                    "assigned_attributes": case_info['assignment'],
                    "vqa_results": vqa_results,
                    "summary": {
                        "accuracy": accuracy,
                        "correct_count": correct_count,
                        "total_questions": total_questions,
                        "average_yes_probability_for_yes_answers": avg_yes_probability, # Modified
                        "average_no_probability_for_no_answers": avg_no_probability # Added
                    }
                }
                all_results.append(result_object)

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not extract ID from filename '{filename}' or process data: {e}")
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
