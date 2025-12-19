import argparse
import json
import os
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering
import torch

PROMPT_DIR = "prompt_generator/prompt" 
OUTPUT_ROOT_DIR = "./" 
RESULT_DIR = "evaluator/vqa_results"

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
    "race": "Is the person {value}?",
    "age": "Is the person {value}?",
    "occupation": "Is the person a {value}?",
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
            # Handle both 'yes' and 'Yes' as possible labels
            if 'yes' in model.config.label2id:
                yes_id = model.config.label2id['yes']
            elif 'Yes' in model.config.label2id:
                yes_id = model.config.label2id['Yes']
            
            yes_probability = probabilities[yes_id].item() if yes_id != -1 else 0.0

            vqa_results[attribute] = {
                "question": question,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "yes_probability": yes_probability
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
    parser = argparse.ArgumentParser(description="Runs n-wise intersectional evaluation using a VQA model.")
    parser.add_argument("-n", type=int, required=True, help="Number for n-wise analysis (e.g., 2 or 3)")
    parser.add_argument("-d", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for evaluation (cuda or cpu)")
    args = parser.parse_args()
    n = args.n
    print(f"--- Starting {n}-wise VQA Evaluation ---")

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
            print(f"--- Processing image: {filename} ---")
            case_id = int(filename.split('_')[0])
            case_info = test_cases.get(case_id)
            if not case_info:
                print(f"Warning: Prompt info for ID {case_id} (from {filename}) not found.")
                continue

            # Generate Question-Answer pairs
            qa_pairs = []
            
            for attribute, value in case_info['assignment'].items():
                template = QUESTION_TEMPLATES.get(attribute, QUESTION_TEMPLATES["default"])
                question = template.format(value=value)
                qa_pairs.append((attribute, question, "yes")) # Attribute, Question, Expected Answer

            full_prompt_question = f"Is the image showing '{case_info['prompt']}'?"
            qa_pairs.append(("full_prompt_check", full_prompt_question, "yes"))

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

                # Calculate average yes_probability
                probabilities = [
                    res["yes_probability"] for attr, res in vqa_results.items() if attr != "full_prompt_check"
                ]
                avg_yes_probability = sum(probabilities) / len(probabilities) if probabilities else 0.0

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
                        "average_yes_probability": avg_yes_probability
                    }
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