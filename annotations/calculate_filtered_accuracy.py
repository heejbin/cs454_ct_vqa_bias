#!/usr/bin/env python3
"""
Filtered Annotation vs VQA Accuracy Analysis Script
특정 속성들에 대해서만 accuracy를 계산합니다:
- age
- appearance_body_status: "Thin", "Fat"만
- disability_status: "without a disability", "who uses a wheelchair"만
- gender
- location
- race_ethnicity

annotation data의 prompt_image_match와 VQA 결과의 yes_probability를 비교하여 accuracy를 측정합니다.
yes_probability 기준: >66% = true, <=33% = false, unsure는 제외하고 True/False만 비교
"""

import json
from collections import defaultdict
from pathlib import Path


def should_include_factor(factor_name, item_value):
    """
    해당 factor와 item_value가 필터링 조건에 맞는지 확인합니다.
    
    Args:
        factor_name: factor 이름 (예: "age", "appearance_body_status", etc.)
        item_value: 해당 factor의 값 (예: "Thin", "Fat", etc.)
    
    Returns:
        bool: 포함해야 하면 True, 아니면 False
    """
    # 항상 포함하는 factor들
    always_include = ["age", "gender", "location", "race_ethnicity"]
    if factor_name in always_include:
        return True
    
    # appearance_body_status: "Thin", "Fat"만 포함
    if factor_name == "appearance_body_status":
        return item_value in ["Thin", "Fat"]
    
    # disability_status: "without a disability", "who uses a wheelchair"만 포함
    if factor_name == "disability_status":
        return item_value in ["without a disability", "who uses a wheelchair"]
    
    # 그 외는 제외
    return False


def compare_annotation_vqa_filtered(annotation_file, vqa_file):
    """
    annotation 파일과 VQA 결과 파일을 비교하여 accuracy를 계산합니다.
    특정 속성들만 필터링하여 계산합니다.
    
    VQA의 yes_probability를 기준으로 채점합니다:
    - >66%: true
    - 33-66%: unsure (비교 제외)
    - <=33%: false
    
    Args:
        annotation_file: annotation JSON 파일 경로
        vqa_file: VQA 결과 JSON 파일 경로
    """
    # JSON 파일 로드
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    with open(vqa_file, 'r', encoding='utf-8') as f:
        vqa_results = json.load(f)
    
    # image_filename을 키로 하는 딕셔너리 생성 (빠른 조회를 위해)
    vqa_dict = {}
    for vqa_item in vqa_results:
        filename = vqa_item.get("image_filename")
        if filename:
            vqa_dict[filename] = vqa_item
    
    # 각 factor별로 accuracy 계산
    # 구조: {factor_name: {"correct": count, "total": count, "unsure_excluded": count, "vqa_missing": count, "filtered_out": count}}
    factor_stats = defaultdict(lambda: {"correct": 0, "total": 0, "unsure_excluded": 0, "vqa_missing": 0, "filtered_out": 0})
    
    # 각 factor별 confusion matrix (2x2: True, False만)
    # 구조: {factor_name: {"true": {"true": 0, "false": 0}, "false": {"true": 0, "false": 0}}}
    factor_confusion = defaultdict(lambda: {
        "true": {"true": 0, "false": 0},
        "false": {"true": 0, "false": 0}
    })
    
    # 전체 통계
    overall_stats = {"correct": 0, "total": 0, "unsure_excluded": 0, "vqa_missing": 0, "filtered_out": 0, "matched_images": 0}
    
    # 전체 confusion matrix (2x2: True, False만)
    overall_confusion = {
        "true": {"true": 0, "false": 0},
        "false": {"true": 0, "false": 0}
    }
    
    # 각 annotation을 순회하며 VQA 결과와 비교
    for annotation in annotations:
        filename = annotation.get("image_filename")
        if not filename:
            continue
        
        # 해당 이미지의 VQA 결과 찾기
        vqa_item = vqa_dict.get(filename)
        if not vqa_item:
            continue
        
        overall_stats["matched_images"] += 1
        vqa_results_data = vqa_item.get("vqa_results", {})
        factor_annotations = annotation.get("factor_annotations", {})
        assigned_attributes = vqa_item.get("assigned_attributes", {})
        
        # annotation에 있는 factor만 순회하여 비교
        for factor_name, factor_data in factor_annotations.items():
            # 필터링: 특정 속성들만 포함
            item_value = assigned_attributes.get(factor_name)
            if not should_include_factor(factor_name, item_value):
                factor_stats[factor_name]["filtered_out"] += 1
                overall_stats["filtered_out"] += 1
                continue
            
            prompt_image_match = factor_data.get("prompt_image_match", "").lower()
            
            # VQA 결과에서 해당 factor 찾기
            vqa_factor = vqa_results_data.get(factor_name)
            if not vqa_factor:
                factor_stats[factor_name]["vqa_missing"] += 1
                overall_stats["vqa_missing"] += 1
                continue
            
            # yes_probability를 기준으로 true/unsure/false 판단
            yes_probability = vqa_factor.get("yes_probability", 0.0)
            
            # yes_probability 기준으로 VQA 결과 분류
            # > 66%: true, <= 33%: false, 33-66%는 unsure로 제외
            if yes_probability > 0.66:
                vqa_result = "true"
            elif yes_probability <= 0.33:
                vqa_result = "false"
            else:
                vqa_result = "unsure"
            
            # accuracy 계산은 True/False만 비교 (Unsure 제외)
            if prompt_image_match == "unsure":
                factor_stats[factor_name]["unsure_excluded"] += 1
                overall_stats["unsure_excluded"] += 1
                continue
            
            # VQA의 unsure도 accuracy 계산에서 제외
            if vqa_result == "unsure":
                factor_stats[factor_name]["unsure_excluded"] += 1
                overall_stats["unsure_excluded"] += 1
                continue
            
            # True/False만 confusion matrix에 기록 (2x2)
            factor_confusion[factor_name][prompt_image_match][vqa_result] += 1
            overall_confusion[prompt_image_match][vqa_result] += 1
            
            factor_stats[factor_name]["total"] += 1
            overall_stats["total"] += 1
            
            # 비교: annotation의 true/false와 VQA의 yes_probability 기반 결과 비교
            if prompt_image_match == "true" and vqa_result == "true":
                # True Positive: 둘 다 true
                factor_stats[factor_name]["correct"] += 1
                overall_stats["correct"] += 1
            elif prompt_image_match == "false" and vqa_result == "false":
                # True Negative: 둘 다 false
                factor_stats[factor_name]["correct"] += 1
                overall_stats["correct"] += 1
    
    # 결과 출력
    print("=" * 90)
    print("Filtered Annotation vs VQA Accuracy Analysis")
    print("(Only selected factors: age, appearance_body_status (Thin/Fat only),")
    print(" disability_status (without a disability/wheelchair only), gender, location, race_ethnicity)")
    print("(VQA scoring based on yes_probability: >66%=true, <=33%=false, unsure excluded)")
    print("(Only True/False comparisons, Unsure cases are excluded)")
    print("=" * 90)
    print(f"Matched images: {overall_stats['matched_images']}")
    print(f"Total comparisons (excluding unsure and missing VQA): {overall_stats['total']}")
    print(f"Excluded (unsure): {overall_stats['unsure_excluded']}")
    print(f"Excluded (VQA missing): {overall_stats['vqa_missing']}")
    print(f"Excluded (filtered out): {overall_stats['filtered_out']}")
    print()
    
    if overall_stats["total"] > 0:
        overall_accuracy = (overall_stats["correct"] / overall_stats["total"]) * 100
        print(f"Overall Accuracy: {overall_stats['correct']}/{overall_stats['total']} = {overall_accuracy:.2f}%")
    print()
    print("=" * 90)
    print()
    
    # Factor별 상세 결과
    sorted_factors = sorted(factor_stats.keys())
    
    print("Factor-wise Accuracy:")
    print("-" * 90)
    print(f"{'Factor':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Unsure':<10} {'VQA Missing':<12} {'Filtered':<10}")
    print("-" * 90)
    
    for factor_name in sorted_factors:
        stats = factor_stats[factor_name]
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"]) * 100
            print(f"{factor_name:<30} {stats['correct']:<10} {stats['total']:<10} {accuracy:>6.2f}%      {stats['unsure_excluded']:<10} {stats['vqa_missing']:<12} {stats['filtered_out']:<10}")
        else:
            print(f"{factor_name:<30} {'N/A':<10} {'0':<10} {'N/A':<12} {stats['unsure_excluded']:<10} {stats['vqa_missing']:<12} {stats['filtered_out']:<10}")
    
    print("=" * 90)
    print()
    
    # 상세 통계 출력
    print("Detailed Statistics:")
    print("-" * 90)
    
    for factor_name in sorted_factors:
        stats = factor_stats[factor_name]
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"]) * 100
            print(f"\nFactor: {factor_name}")
            print(f"  Correct: {stats['correct']}/{stats['total']} ({accuracy:.2f}%)")
            print(f"  Unsure excluded: {stats['unsure_excluded']}")
            print(f"  VQA missing: {stats['vqa_missing']}")
            print(f"  Filtered out: {stats['filtered_out']}")
        else:
            print(f"\nFactor: {factor_name}")
            print(f"  No comparisons (all were unsure, VQA missing, or filtered out)")
            print(f"  Unsure excluded: {stats['unsure_excluded']}")
            print(f"  VQA missing: {stats['vqa_missing']}")
            print(f"  Filtered out: {stats['filtered_out']}")
    
    print()
    print("=" * 90)
    print()
    
    # Confusion Matrix 출력 (2x2: True, False만)
    print("=" * 90)
    print("Confusion Matrix (2x2: True, False only)")
    print("=" * 90)
    print()
    
    # 전체 Confusion Matrix
    print("Overall Confusion Matrix:")
    print("-" * 90)
    print(f"{'':<20} {'VQA: True':<15} {'VQA: False':<15}")
    print("-" * 90)
    print(f"{'Annotation: True':<20} {overall_confusion['true']['true']:<15} {overall_confusion['true']['false']:<15}")
    print(f"{'Annotation: False':<20} {overall_confusion['false']['true']:<15} {overall_confusion['false']['false']:<15}")
    print("-" * 90)
    
    # 전체 지표 계산 (True/False만 사용)
    total_cm = (overall_confusion["true"]["true"] + overall_confusion["true"]["false"] + 
                overall_confusion["false"]["true"] + overall_confusion["false"]["false"])
    if total_cm > 0:
        tp = overall_confusion["true"]["true"]
        fp = overall_confusion["false"]["true"]
        tn = overall_confusion["false"]["false"]
        fn = overall_confusion["true"]["false"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nOverall Metrics (True/False only):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall (Sensitivity): {recall:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        print(f"  Specificity: {specificity:.4f}")
    
    print()
    print("=" * 90)
    print()
    
    # Factor별 Confusion Matrix
    print("Factor-wise Confusion Matrices:")
    print("=" * 90)
    
    for factor_name in sorted_factors:
        cm = factor_confusion[factor_name]
        total_factor = (cm["true"]["true"] + cm["true"]["false"] + 
                        cm["false"]["true"] + cm["false"]["false"])
        
        if total_factor > 0:
            print(f"\nFactor: {factor_name}")
            print("-" * 90)
            print(f"{'':<20} {'VQA: True':<15} {'VQA: False':<15}")
            print("-" * 90)
            print(f"{'Annotation: True':<20} {cm['true']['true']:<15} {cm['true']['false']:<15}")
            print(f"{'Annotation: False':<20} {cm['false']['true']:<15} {cm['false']['false']:<15}")
            print("-" * 90)
            
            # Factor별 지표 계산 (True/False만 사용)
            tp = cm["true"]["true"]
            fp = cm["false"]["true"]
            tn = cm["false"]["false"]
            fn = cm["true"]["false"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"  Metrics (True/False only):")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall (Sensitivity): {recall:.4f}")
            print(f"    F1-Score: {f1_score:.4f}")
            print(f"    Specificity: {specificity:.4f}")
        else:
            print(f"\nFactor: {factor_name}")
            print("  No confusion matrix data (all were missing or filtered)")
    
    print()
    print("=" * 90)
    
    return factor_stats, overall_stats, factor_confusion, overall_confusion


if __name__ == "__main__":
    # 파일 경로 설정
    script_dir = Path("/root/cs454_ct_diffusion_bias/annotations")
    annotation_file = script_dir / "total_annotation.json"
    vqa_file = Path("/root/cs454_ct_diffusion_bias/evaluator/vqa_results/scores_2wise.json")
    
    if not annotation_file.exists():
        print(f"Error: {annotation_file} 파일을 찾을 수 없습니다.")
        exit(1)
    
    if not vqa_file.exists():
        print(f"Error: {vqa_file} 파일을 찾을 수 없습니다.")
        exit(1)
    
    compare_annotation_vqa_filtered(annotation_file, vqa_file)

