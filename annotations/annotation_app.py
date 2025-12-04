#!/usr/bin/env python3
"""
Interactive annotation system for prompt-image-VQA matching evaluation.
"""

import json
import os
import shutil
from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path

# Paths (defined before Flask app initialization)
BASE_DIR = Path(__file__).parent.parent  # /root/cs454_ct_diffusion_bias
ANNOTATIONS_DIR = BASE_DIR / "annotations"  # /root/cs454_ct_diffusion_bias/annotations

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
VQA_RESULTS_PATH = BASE_DIR / "evaluator" / "vqa_results" / "scores_2wise.json"
PROMPTS_PATH = BASE_DIR / "prompt_generator" / "prompt" / "ct_prompts_2wise.json"
CONFIG_PATH = BASE_DIR / "prompt_generator" / "prompt_config" / "ct_config_2wise.jsonc"
IMAGES_DIR = BASE_DIR / "outputs_2wise"
def get_progress_path(annotator):
    """Get progress file path for a specific annotator."""
    if annotator:
        return ANNOTATIONS_DIR / f"annotation_progress_{annotator}.json"
    return ANNOTATIONS_DIR / "annotation_progress.json"

def get_annotations_path(annotator):
    """Get annotations file path for a specific annotator."""
    if annotator:
        return ANNOTATIONS_DIR / f"annotations_{annotator}.json"
    return ANNOTATIONS_DIR / "annotations.json"

def save_annotations_safely(annotations_path, annotations_data):
    """Save annotations file safely with backup and atomic write."""
    try:
        # Count current annotations
        annotation_count = len(annotations_data)
        
        # Create backup if file exists
        backup_path = annotations_path.with_suffix('.json.bak')
        if annotations_path.exists():
            shutil.copy2(annotations_path, backup_path)
        
        # Write to temporary file first
        temp_path = annotations_path.with_suffix('.json.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(annotations_data, f, indent=2, ensure_ascii=False)
        
        # Atomically replace the original file
        temp_path.replace(annotations_path)
        
        # Create milestone copy every 50 annotations
        if annotation_count > 0 and annotation_count % 50 == 0:
            milestone_path = annotations_path.parent / f"{annotations_path.stem}_{annotation_count}.json"
            shutil.copy2(annotations_path, milestone_path)
            print(f"Created milestone copy: {milestone_path}")
        
        # Remove backup if save was successful
        if backup_path.exists():
            backup_path.unlink()
        
        return True
    except Exception as e:
        # Restore from backup if save failed
        backup_path = annotations_path.with_suffix('.json.bak')
        if backup_path.exists():
            try:
                shutil.copy2(backup_path, annotations_path)
                print(f"Restored annotations from backup due to error: {e}")
            except:
                pass
        print(f"Error saving annotations: {e}")
        return False

# Load data
print("Loading data...")
with open(VQA_RESULTS_PATH, 'r', encoding='utf-8') as f:
    vqa_data = json.load(f)

with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
    prompts_data = json.load(f)

# Load config to get factors information
import re
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_content = f.read()
        # Remove comments (jsonc format)
        config_content = re.sub(r'//.*', '', config_content)
        config_data = json.loads(config_content)
except Exception as e:
    print(f"Warning: Could not load config file: {e}")
    # Fallback: use factors from prompts data if available
    config_data = {"factors": []}

# Hyperparameter: number of annotations per prompt
N_PER_PARAMETER = 10

# Annotators
ANNOTATORS = ["장현", "희진", "승재", "준영"]
# Prompt 0 is shared by all annotators
# Remaining prompts (1-68) are distributed among 4 annotators, 17 each
# Distribution: 장현(1-17), 희진(18-34), 승재(35-51), 준영(52-68)
PROMPT_ASSIGNMENTS = {
    "장현": [0] + list(range(1, 18)),      # 0, 1-17
    "희진": [0] + list(range(18, 35)),     # 0, 18-34
    "승재": [0] + list(range(35, 52)),     # 0, 35-51
    "준영": [0] + list(range(52, 69))      # 0, 52-68
}

# Create prompt lookup by full_prompt and by ID
prompt_lookup = {case['prompt']: case for case in prompts_data['test_cases']}
prompt_id_lookup = {case['id']: case['prompt'] for case in prompts_data['test_cases']}

# Get factors list (ordered)
if config_data.get('factors'):
    FACTORS = [factor['name'] for factor in config_data['factors']]
    FACTOR_INFO = {factor['name']: factor for factor in config_data['factors']}
else:
    # Fallback: extract from first sample's assigned_attributes
    if vqa_data:
        FACTORS = list(vqa_data[0]['assigned_attributes'].keys())
        FACTOR_INFO = {name: {'name': name, 'attribute': '', 'labels': []} for name in FACTORS}
    else:
        FACTORS = []
        FACTOR_INFO = {}

# Group samples by prompt and select first N_PER_PARAMETER samples for each prompt
def group_samples_by_prompt():
    """Group samples by prompt and return indices of first N_PER_PARAMETER samples for each prompt, sorted by filename."""
    prompt_groups = {}
    for idx, sample in enumerate(vqa_data):
        full_prompt = sample['full_prompt']
        if full_prompt not in prompt_groups:
            prompt_groups[full_prompt] = []
        prompt_groups[full_prompt].append(idx)
    
    # Select first N_PER_PARAMETER indices for each prompt, sorted by filename (numeric order)
    selected_indices = []
    for prompt, indices in prompt_groups.items():
        # Sort indices by filename with numeric ordering
        # Extract the number from filename (e.g., "0_0.png" -> 0, "0_10.png" -> 10)
        def get_image_number(idx):
            filename = vqa_data[idx]['image_filename']
            try:
                # Extract number after underscore (e.g., "0_10.png" -> "10")
                if '_' in filename:
                    num_str = filename.split('_')[1].split('.')[0]
                    return int(num_str)
            except:
                pass
            return 0
        
        sorted_indices = sorted(indices, key=get_image_number)
        # Take first N_PER_PARAMETER
        selected_indices.extend(sorted_indices[:N_PER_PARAMETER])
    
    # Sort by original order
    selected_indices.sort()
    return set(selected_indices), prompt_groups

# Get selected sample indices (first N_PER_PARAMETER for each prompt)
selected_sample_indices, prompt_groups = group_samples_by_prompt()

# Create prompt ID to full_prompt mapping from vqa_data
prompt_id_to_full_prompt = {}
for sample in vqa_data:
    full_prompt = sample['full_prompt']
    # Find prompt ID by matching with prompt_lookup
    for prompt_id, prompt_text in prompt_id_lookup.items():
        if prompt_text == full_prompt:
            prompt_id_to_full_prompt[prompt_id] = full_prompt
            break

# Progress will be loaded per annotator when they select their name
progress = {"current_index": 0, "completed": set(), "annotator": None}

# Don't load annotations on startup - will be loaded when annotator is selected
annotations = []

print(f"Loaded {len(vqa_data)} samples")
print(f"Current progress: {progress['current_index']}/{len(vqa_data)}")


@app.route('/')
def index():
    """Main annotation interface."""
    return render_template('index.html')


@app.route('/api/annotators')
def get_annotators():
    """Get list of annotators."""
    return jsonify({
        "annotators": ANNOTATORS,
        "prompt_assignments": PROMPT_ASSIGNMENTS
    })


@app.route('/api/set_annotator', methods=['POST'])
def set_annotator():
    """Set current annotator."""
    data = request.json
    annotator = data.get('annotator')
    
    if annotator not in ANNOTATORS:
        return jsonify({"error": "Invalid annotator"}), 400
    
    # Store annotator in progress
    progress['annotator'] = annotator
    
    # Load annotations for this annotator
    global annotations
    annotations_path = get_annotations_path(annotator)
    if annotations_path.exists():
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    else:
        annotations = []
    
    # Get assigned prompt IDs for this annotator
    assigned_prompt_ids = PROMPT_ASSIGNMENTS.get(annotator, [])
    
    # Get full prompts for assigned prompt IDs
    assigned_prompts = set()
    for prompt_id in assigned_prompt_ids:
        if prompt_id in prompt_id_to_full_prompt:
            assigned_prompts.add(prompt_id_to_full_prompt[prompt_id])
    
    # Filter selected_sample_indices to only include samples with assigned prompts
    user_selected_indices = []
    for idx in selected_sample_indices:
        sample = vqa_data[idx]
        if sample['full_prompt'] in assigned_prompts:
            user_selected_indices.append(idx)
    
    user_selected_indices.sort()
    
    # Update current_index to first assigned sample
    if user_selected_indices:
        progress['current_index'] = user_selected_indices[0]
    else:
        progress['current_index'] = len(vqa_data)
    
    # Calculate completed samples from annotations
    completed_from_annotations = set()
    for ann in annotations:
        sample_id = f"{ann.get('image_id')}_{ann.get('image_filename')}"
        completed_from_annotations.add(sample_id)
    
    # Merge completed sets
    progress['completed'] = progress['completed'] | completed_from_annotations
    
    # Update current_index to first uncompleted assigned sample
    if user_selected_indices:
        # Find first uncompleted sample
        for idx in user_selected_indices:
            sample = vqa_data[idx]
            sample_id = f"{sample['image_id']}_{sample['image_filename']}"
            if sample_id not in progress['completed']:
                progress['current_index'] = idx
                break
        else:
            # All samples completed
            progress['current_index'] = len(vqa_data)
    else:
        progress['current_index'] = len(vqa_data)
    
    # Save progress for this annotator
    progress_path = get_progress_path(annotator)
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump({
            "current_index": progress['current_index'],
            "completed": list(progress['completed']),
            "annotator": annotator
        }, f, indent=2)
    
    return jsonify({
        "success": True,
        "annotator": annotator,
        "assigned_prompt_ids": assigned_prompt_ids,
        "total_assigned_samples": len(user_selected_indices)
    })


@app.route('/api/current_sample')
def get_current_sample():
    """Get current sample to annotate."""
    # Always require annotator selection - don't use saved annotator from progress
    # This ensures users select their name on each new session
    if not progress.get('annotator'):
        return jsonify({
            "annotator_required": True,
            "annotators": ANNOTATORS
        })
    
    annotator = progress['annotator']
    assigned_prompt_ids = PROMPT_ASSIGNMENTS.get(annotator, [])
    assigned_prompts = set()
    for prompt_id in assigned_prompt_ids:
        if prompt_id in prompt_id_lookup:
            assigned_prompts.add(prompt_id_lookup[prompt_id])
    
    current_idx = progress['current_index']
    
    # Skip samples that are not in selected_indices or not assigned to this annotator
    max_iterations = len(vqa_data) * 2  # Safety limit
    iterations = 0
    
    while current_idx < len(vqa_data) and iterations < max_iterations:
        sample = vqa_data[current_idx]
        
        # Check if current sample is in selected indices and assigned to this annotator
        if current_idx not in selected_sample_indices or sample['full_prompt'] not in assigned_prompts:
            # Skip this sample and move to next assigned sample
            remaining_indices = [
                idx for idx in selected_sample_indices 
                if idx > current_idx and vqa_data[idx]['full_prompt'] in assigned_prompts
            ]
            if remaining_indices:
                current_idx = min(remaining_indices)
                progress['current_index'] = current_idx
                iterations += 1
                continue
            else:
                # No more assigned samples
                break
        
        # Found a valid assigned sample
        break
    
    # Save progress if we skipped any samples
    if iterations > 0:
        progress_path = get_progress_path(annotator)
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({
                "current_index": progress['current_index'],
                "completed": list(progress['completed']),
                "annotator": annotator
            }, f, indent=2)
    
    sample = vqa_data[current_idx] if current_idx < len(vqa_data) else None
    if current_idx >= len(vqa_data) or current_idx not in selected_sample_indices or (sample and sample['full_prompt'] not in assigned_prompts):
        return jsonify({
            "completed": True,
            "message": "All assigned samples have been annotated!"
        })
    
    sample = vqa_data[current_idx]
    
    # Get prompt info
    full_prompt = sample['full_prompt']
    prompt_info = prompt_lookup.get(full_prompt, {})
    
    # Check if already annotated
    sample_id = f"{sample['image_id']}_{sample['image_filename']}"
    is_completed = sample_id in progress['completed']
    
    # Get existing annotations if any
    existing_annotations = None
    for ann in annotations:
        if ann.get('image_id') == sample['image_id'] and ann.get('image_filename') == sample['image_filename']:
            existing_annotations = ann
            break
    
    # Count how many samples of this prompt have been annotated
    prompt_indices = prompt_groups.get(full_prompt, [])
    annotated_count = sum(1 for idx in prompt_indices if f"{vqa_data[idx]['image_id']}_{vqa_data[idx]['image_filename']}" in progress['completed'])
    prompt_count = annotated_count
    
    # Extract image number from filename (e.g., "0_0.png" -> 0, "0_1.png" -> 1)
    image_filename = sample['image_filename']
    image_number = 0
    try:
        if '_' in image_filename:
            image_number = int(image_filename.split('_')[1].split('.')[0])
    except:
        pass
    
    # Prepare factors for annotation
    factors_for_annotation = []
    for factor_name in FACTORS:
        assigned_value = sample['assigned_attributes'].get(factor_name, '')
        vqa_result = sample['vqa_results'].get(factor_name, {})
        factor_info = FACTOR_INFO.get(factor_name, {})
        
        # Get all labels for this factor
        all_labels = factor_info.get('labels', [])
        other_labels = [label for label in all_labels if label != assigned_value]
        
        factors_for_annotation.append({
            "name": factor_name,
            "assigned_value": assigned_value,
            "other_labels": other_labels,
            "vqa_question": vqa_result.get('question', ''),
            "vqa_answer": vqa_result.get('predicted_answer', ''),
            "vqa_probability": vqa_result.get('yes_probability', 0),
            "attribute_type": factor_info.get('attribute', ''),
            "labels": all_labels
        })
    
    return jsonify({
        "completed": False,
        "index": current_idx,
        "total": len(vqa_data),
        "sample": {
            "image_id": sample['image_id'],
            "image_filename": sample['image_filename'],
            "image_number": image_number,
            "full_prompt": full_prompt,
            "prompt_info": prompt_info,
            "vqa_results": sample['vqa_results'],
            "assigned_attributes": sample['assigned_attributes'],
            "factors": factors_for_annotation
        },
        "is_completed": is_completed,
        "existing_annotations": existing_annotations,
        "prompt_annotation_count": prompt_count,
        "n_per_parameter": N_PER_PARAMETER,
        "can_annotate": prompt_count < N_PER_PARAMETER
    })


@app.route('/api/save_partial', methods=['POST'])
def save_partial_annotation():
    """Save partial annotation (called after each factor is annotated)."""
    data = request.json
    
    sample_idx = progress['current_index']
    if sample_idx >= len(vqa_data):
        return jsonify({"error": "All samples completed"}), 400
    
    sample = vqa_data[sample_idx]
    sample_id = f"{sample['image_id']}_{sample['image_filename']}"
    full_prompt = sample['full_prompt']
    
    annotator = progress.get('annotator')
    if not annotator:
        return jsonify({"error": "Annotator not set"}), 400
    
    # Load current annotations
    annotations_path = get_annotations_path(annotator)
    if annotations_path.exists():
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    else:
        annotations = []
    
    # Find existing annotation or create new one
    annotation = None
    for ann in annotations:
        if ann.get('image_id') == sample['image_id'] and ann.get('image_filename') == sample['image_filename']:
            annotation = ann
            break
    
    if not annotation:
        annotation = {
            "image_id": sample['image_id'],
            "image_filename": sample['image_filename'],
            "full_prompt": full_prompt,
            "factor_annotations": {},
            "image_has_anomalies": None,
            "timestamp": data.get('timestamp')
        }
        annotations.append(annotation)
    
    # Update annotation with current data
    annotation['factor_annotations'] = data.get('factor_annotations', {})
    annotation['image_has_anomalies'] = data.get('image_has_anomalies', None)
    if data.get('timestamp'):
        annotation['timestamp'] = data.get('timestamp')
    
    # Save annotations to annotator-specific file (safely with backup)
    if not save_annotations_safely(annotations_path, annotations):
        return jsonify({"error": "Failed to save annotations"}), 500
    
    return jsonify({"success": True, "message": "Partial annotation saved"})


@app.route('/api/annotate', methods=['POST'])
def save_annotation():
    """Save annotation for current sample."""
    data = request.json
    
    sample_idx = progress['current_index']
    if sample_idx >= len(vqa_data):
        return jsonify({"error": "All samples completed"}), 400
    
    sample = vqa_data[sample_idx]
    sample_id = f"{sample['image_id']}_{sample['image_filename']}"
    full_prompt = sample['full_prompt']
    
    # Check if this sample is in selected indices
    if sample_idx not in selected_sample_indices:
        # Find next selected sample
        remaining_indices = [idx for idx in selected_sample_indices if idx > sample_idx]
        if remaining_indices:
            progress['current_index'] = min(remaining_indices)
        else:
            progress['current_index'] = len(vqa_data)
        
        annotator = progress.get('annotator')
        if annotator:
            progress_path = get_progress_path(annotator)
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "current_index": progress['current_index'],
                    "completed": list(progress['completed']),
                    "annotator": annotator
                }, f, indent=2)
        return jsonify({
            "success": False,
            "error": "This sample is not in the selected set",
            "next_index": progress['current_index'],
            "total": len(selected_sample_indices)
        })
    
    annotator = progress.get('annotator')
    if not annotator:
        return jsonify({"error": "Annotator not set"}), 400
    
    # Load current annotations
    annotations_path = get_annotations_path(annotator)
    if annotations_path.exists():
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    else:
        annotations = []
    
    # Check if this is a new annotation or updating existing one
    is_new_annotation = True
    for ann in annotations:
        if ann.get('image_id') == sample['image_id'] and ann.get('image_filename') == sample['image_filename']:
            is_new_annotation = False
            break
    
    # Create annotation entry with factor-level annotations
    # factor_annotations structure: {factor_name: {"prompt_image_match": "true"/"false"/"unsure", "image_vqa_match": "true"/"false"/"unsure"}}
    annotation = {
        "image_id": sample['image_id'],
        "image_filename": sample['image_filename'],
        "full_prompt": full_prompt,
        "factor_annotations": data.get('factor_annotations', {}),
        "image_has_anomalies": data.get('image_has_anomalies'),  # "true" or "false" or null
        "timestamp": data.get('timestamp')
    }
    
    # Remove old annotation if exists
    annotations[:] = [a for a in annotations if not (
        a.get('image_id') == sample['image_id'] and 
        a.get('image_filename') == sample['image_filename']
    )]
    
    # Add new annotation
    annotations.append(annotation)
    
    # Save annotations to annotator-specific file (safely with backup)
    if not save_annotations_safely(annotations_path, annotations):
        return jsonify({"error": "Failed to save annotations"}), 500
    
    # Mark as completed
    progress['completed'].add(sample_id)
    
    # Move to next assigned sample
    annotator = progress.get('annotator')
    if annotator:
        assigned_prompt_ids = PROMPT_ASSIGNMENTS.get(annotator, [])
        assigned_prompts = set()
        for prompt_id in assigned_prompt_ids:
            if prompt_id in prompt_id_lookup:
                assigned_prompts.add(prompt_id_lookup[prompt_id])
        
        remaining_indices = [
            idx for idx in selected_sample_indices 
            if idx > sample_idx and vqa_data[idx]['full_prompt'] in assigned_prompts
        ]
    else:
        remaining_indices = [idx for idx in selected_sample_indices if idx > sample_idx]
    
    if remaining_indices:
        progress['current_index'] = min(remaining_indices)
    else:
        progress['current_index'] = len(vqa_data)
    
    # Save progress
    annotator = progress.get('annotator')
    if annotator:
        progress_path = get_progress_path(annotator)
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({
                "current_index": progress['current_index'],
                "completed": list(progress['completed']),
                "annotator": annotator
            }, f, indent=2)
    
    # Count annotated samples for this prompt
    prompt_indices = prompt_groups.get(full_prompt, [])
    annotated_count = sum(1 for idx in prompt_indices if f"{vqa_data[idx]['image_id']}_{vqa_data[idx]['image_filename']}" in progress['completed'])
    
    return jsonify({
        "success": True,
        "next_index": progress['current_index'],
        "total": len(selected_sample_indices),
        "prompt_annotation_count": annotated_count,
        "n_per_parameter": N_PER_PARAMETER
    })


@app.route('/api/skip', methods=['POST'])
def skip_sample():
    """Skip current sample without annotating."""
    if not progress.get('annotator'):
        return jsonify({"error": "Annotator not set"}), 400
    
    annotator = progress['annotator']
    assigned_prompt_ids = PROMPT_ASSIGNMENTS.get(annotator, [])
    assigned_prompts = set()
    for prompt_id in assigned_prompt_ids:
        if prompt_id in prompt_id_lookup:
            assigned_prompts.add(prompt_id_lookup[prompt_id])
    
    current_idx = progress['current_index']
    
    # Find next assigned sample
    remaining_indices = [
        idx for idx in selected_sample_indices 
        if idx > current_idx and vqa_data[idx]['full_prompt'] in assigned_prompts
    ]
    if remaining_indices:
        progress['current_index'] = min(remaining_indices)
    else:
        progress['current_index'] = len(vqa_data)
    
    progress_path = get_progress_path(annotator)
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump({
            "current_index": progress['current_index'],
            "completed": list(progress['completed']),
            "annotator": annotator
        }, f, indent=2)
    
    return jsonify({
        "success": True,
        "next_index": progress['current_index'],
        "total": len(selected_sample_indices)
    })


@app.route('/api/go_to', methods=['POST'])
def go_to_index():
    """Jump to specific sample index."""
    data = request.json
    target_idx = data.get('index', 0)
    
    if 0 <= target_idx < len(vqa_data):
        progress['current_index'] = target_idx
        
        annotator = progress.get('annotator')
        if annotator:
            progress_path = get_progress_path(annotator)
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "current_index": progress['current_index'],
                    "completed": list(progress['completed']),
                    "annotator": annotator
                }, f, indent=2)
        
        return jsonify({
            "success": True,
            "current_index": progress['current_index']
        })
    else:
        return jsonify({"error": "Invalid index"}), 400


@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from outputs_2wise directory."""
    return send_from_directory(str(IMAGES_DIR), filename)


@app.route('/api/stats')
def get_stats():
    """Get annotation statistics."""
    total = len(selected_sample_indices)
    completed = len(progress['completed'])
    annotated_count = len(annotations)
    
    # Count unique prompts
    unique_prompts = len(set(ann.get('full_prompt', '') for ann in annotations))
    
    # Count prompts that reached limit
    prompts_at_limit = 0
    for prompt, indices in prompt_groups.items():
        selected_for_prompt = [idx for idx in indices if idx in selected_sample_indices]
        annotated_for_prompt = sum(1 for idx in selected_for_prompt if f"{vqa_data[idx]['image_id']}_{vqa_data[idx]['image_filename']}" in progress['completed'])
        if annotated_for_prompt >= N_PER_PARAMETER:
            prompts_at_limit += 1
    
    return jsonify({
        "total_samples": total,
        "completed_samples": completed,
        "annotated_samples": annotated_count,
        "current_index": progress['current_index'],
        "progress_percent": round((completed / total * 100) if total > 0 else 0, 2),
        "unique_prompts_annotated": unique_prompts,
        "prompts_at_limit": prompts_at_limit,
        "n_per_parameter": N_PER_PARAMETER
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = BASE_DIR / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Annotation System Starting...")
    print("="*60)
    print(f"Total samples: {len(vqa_data)}")
    print(f"Selected samples (first {N_PER_PARAMETER} per prompt): {len(selected_sample_indices)}")
    print(f"Current index: {progress['current_index']}")
    print(f"Completed: {len(progress['completed'])}")
    print(f"n_per_parameter: {N_PER_PARAMETER}")
    print(f"Unique prompts: {len(prompt_groups)}")
    print("\nTo access the interface:")
    print("1. Set up SSH tunnel: ssh -L 5000:localhost:5000 user@vm-ip")
    print("2. Open browser: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

