import argparse
import json
import os
from functools import partial
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
DEFAULT_BATCH_SIZE = 16

QUESTION_TEMPLATES = {
    "gender": "Is the person's gender {value}?",
    "race": "Is the person {value}?",
    "age": "Is the person {value}?",
    "occupation": "Is the person a {value}?",
    "default": "Is the person {value}?" # Fallback template
}


class VQADataset(Dataset):
    """Dataset for VQA evaluation that flattens all image-question pairs."""
    
    def __init__(self, samples):
        """
        Args:
            samples: List of dicts with keys:
                - image_path: path to image file
                - question: question string
                - attribute: attribute name
                - expected_answer: expected answer string
                - case_id: test case ID
                - filename: original filename
                - case_info: full case info dict
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert("RGB")
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new("RGB", (224, 224), color="black")  # Placeholder
        
        return {
            'image': image,
            'question': sample['question'],
            'attribute': sample['attribute'],
            'expected_answer': sample['expected_answer'],
            'case_id': sample['case_id'],
            'filename': sample['filename'],
            'case_info': sample['case_info']
        }


def collate_fn(batch, processor):
    """Custom collate function that uses the processor for batching."""
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    
    # Process images and questions together
    encoding = processor(images, questions, return_tensors="pt", padding=True)
    
    # Keep metadata
    metadata = [{
        'attribute': item['attribute'],
        'expected_answer': item['expected_answer'],
        'case_id': item['case_id'],
        'filename': item['filename'],
        'case_info': item['case_info'],
        'question': item['question']
    } for item in batch]
    
    return encoding, metadata


def evaluate_batch(model, encoding, metadata, device):
    """Evaluate a batch of image-question pairs."""
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    logits = outputs.logits
    predictions = logits.argmax(dim=-1)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get yes_id once
    yes_id = model.config.label2id.get('yes', model.config.label2id.get('Yes', -1))
    
    results = []
    for i, meta in enumerate(metadata):
        idx = predictions[i].item()
        predicted_answer = model.config.id2label[idx]
        yes_probability = probabilities[i, yes_id].item() if yes_id != -1 else 0.0
        
        results.append({
            'attribute': meta['attribute'],
            'question': meta['question'],
            'predicted_answer': predicted_answer,
            'is_correct': predicted_answer.lower() == meta['expected_answer'].lower(),
            'yes_probability': yes_probability,
            'case_id': meta['case_id'],
            'filename': meta['filename'],
            'case_info': meta['case_info']
        })
    
    return results

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
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of workers for DataLoader (default: 4)")
    args = parser.parse_args()
    n = args.n
    batch_size = args.batch_size
    num_workers = args.num_workers
    print(f"--- Starting {n}-wise VQA Evaluation ---")
    print(f"Using batch_size={batch_size}, num_workers={num_workers}")

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
    print(f"\nEvaluating images in '{image_dir}' folder...")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_samples = []
    for filename in image_files:
        try:
            case_id = int(filename.split('_')[0])
            case_info = test_cases.get(case_id)
            if not case_info:
                print(f"Warning: Prompt info for ID {case_id} (from {filename}) not found.")
                continue

            image_path = os.path.join(image_dir, filename)
            for attribute, value in case_info['assignment'].items():
                template = QUESTION_TEMPLATES.get(attribute, QUESTION_TEMPLATES["default"])
                question = template.format(value=value)
                all_samples.append({
                    'image_path': image_path,
                    'question': question,
                    'attribute': attribute,
                    'expected_answer': 'yes',
                    'case_id': case_id,
                    'filename': filename,
                    'case_info': case_info
                })

            # full_prompt_question = f"Is the image showing '{case_info['prompt']}'?"
            # all_samples.append({
            #     'image_path': image_path,
            #     'question': full_prompt_question,
            #     'attribute': 'full_prompt_check',
            #     'expected_answer': 'yes',
            #     'case_id': case_id,
            #     'filename': filename,
            #     'case_info': case_info
            # })
        except (ValueError, IndexError):
            print(f"Warning: Could not extract ID from filename '{filename}'")

    if not all_samples:
        print("\nNo valid samples found for evaluation.")
        return

    print(f"Total samples to evaluate: {len(all_samples)}")

    # Create DataLoader
    dataset = VQADataset(all_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, processor=processor),
        pin_memory=(device == "cuda")
    )

    # Process batches
    batch_results = []
    for encoding, metadata in tqdm(dataloader, desc="Processing batches"):
        results = evaluate_batch(model, encoding, metadata, device)
        batch_results.extend(results)

    # Aggregate results by image
    results_by_image = {}
    for result in batch_results:
        key = (result['case_id'], result['filename'])
        if key not in results_by_image:
            results_by_image[key] = {
                'case_id': result['case_id'],
                'filename': result['filename'],
                'case_info': result['case_info'],
                'vqa_results': {}
            }
        results_by_image[key]['vqa_results'][result['attribute']] = {
            'question': result['question'],
            'predicted_answer': result['predicted_answer'],
            'is_correct': result['is_correct'],
            'yes_probability': result['yes_probability']
        }

    # Build final results
    all_results = []
    for (case_id, filename), data in results_by_image.items():
        vqa_results = data['vqa_results']
        case_info = data['case_info']

        correct_count = sum(1 for res in vqa_results.values() if res["is_correct"])
        total_questions = len(vqa_results)
        accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0

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