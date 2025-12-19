#!/usr/bin/env python3
"""
Inter-Annotator Agreement (IAA) Calculation Script
4명의 annotator(희진, 준영, 장현, 승재)의 annotation 결과를 비교하여 IAA를 계산합니다.
"""

import json
from collections import defaultdict, Counter
from pathlib import Path
from itertools import combinations
import numpy as np


def should_include_factor(factor_name, item_value):
    """
    해당 factor와 item_value가 필터링 조건에 맞는지 확인합니다.
    """
    always_include = ["age", "gender", "location", "race_ethnicity"]
    if factor_name in always_include:
        return True
    if factor_name == "appearance_body_status":
        return item_value in ["Thin", "Fat"]
    if factor_name == "disability_status":
        return item_value in ["without a disability", "who uses a wheelchair"]
    return False


def load_annotations(annotation_files):
    """
    여러 annotation 파일을 로드합니다.
    
    Returns:
        dict: {image_filename: {annotator_name: {factor_name: value}}}
    """
    annotators = ["희진", "준영", "장현", "승재"]
    all_annotations = {}
    
    for annotator, filepath in zip(annotators, annotation_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        for annotation in annotations:
            filename = annotation.get("image_filename")
            if not filename:
                continue
            
            if filename not in all_annotations:
                all_annotations[filename] = {}
            
            factor_annotations = annotation.get("factor_annotations", {})
            all_annotations[filename][annotator] = factor_annotations
    
    return all_annotations


def calculate_agreement(annotations_dict, vqa_file=None):
    """
    IAA를 계산합니다.
    
    Args:
        annotations_dict: {image_filename: {annotator_name: {factor_name: value}}}
        vqa_file: VQA 결과 파일 (assigned_attributes를 가져오기 위해)
    """
    # VQA 파일에서 assigned_attributes 가져오기
    vqa_dict = {}
    if vqa_file and Path(vqa_file).exists():
        with open(vqa_file, 'r', encoding='utf-8') as f:
            vqa_results = json.load(f)
        for vqa_item in vqa_results:
            filename = vqa_item.get("image_filename")
            if filename:
                vqa_dict[filename] = vqa_item.get("assigned_attributes", {})
    
    # 각 factor별 통계
    factor_stats = defaultdict(lambda: {
        "total_comparisons": 0,
        "perfect_agreement": 0,  # 4명 모두 일치
        "three_agree": 0,  # 3명 일치
        "two_agree": 0,  # 2명 일치
        "no_agreement": 0,  # 모두 다름
        "pairwise_agreements": []  # pairwise agreement 저장
    })
    
    annotators = ["희진", "준영", "장현", "승재"]
    
    # 각 이미지에 대해 비교 (image_id가 0인 것만)
    for filename, annotator_data in annotations_dict.items():
        if len(annotator_data) != 4:
            continue  # 4명 모두 평가한 경우만
        
        # image_id가 0인 것만 필터링
        # filename에서 image_id 추출 (예: "0_0.png" -> image_id = 0)
        try:
            image_id = int(filename.split('_')[0])
            if image_id != 0:
                continue
        except (ValueError, IndexError):
            continue
        
        assigned_attributes = vqa_dict.get(filename, {})
        
        # 각 factor에 대해 비교
        all_factors = set()
        for annotator in annotators:
            if annotator in annotator_data:
                all_factors.update(annotator_data[annotator].keys())
        
        for factor_name in all_factors:
            # 필터링 적용
            item_value = assigned_attributes.get(factor_name)
            if not should_include_factor(factor_name, item_value):
                continue
            
            # 4명의 평가 수집
            values = []
            for annotator in annotators:
                if annotator in annotator_data:
                    factor_data = annotator_data[annotator].get(factor_name, {})
                    value = factor_data.get("prompt_image_match", "").lower()
                    if value in ["true", "false", "unsure"]:
                        values.append(value)
            
            if len(values) != 4:
                continue  # 4명 모두 평가한 경우만
            
            factor_stats[factor_name]["total_comparisons"] += 1
            
            # 값 분포 확인
            value_counts = Counter(values)
            max_count = max(value_counts.values())
            
            if max_count == 4:
                factor_stats[factor_name]["perfect_agreement"] += 1
            elif max_count == 3:
                factor_stats[factor_name]["three_agree"] += 1
            elif max_count == 2:
                factor_stats[factor_name]["two_agree"] += 1
            else:
                factor_stats[factor_name]["no_agreement"] += 1
            
            # Pairwise agreement 계산
            for i, j in combinations(range(4), 2):
                if values[i] == values[j]:
                    factor_stats[factor_name]["pairwise_agreements"].append(1)
                else:
                    factor_stats[factor_name]["pairwise_agreements"].append(0)
    
    return factor_stats


def calculate_fleiss_kappa(factor_stats):
    """
    Fleiss' Kappa를 계산합니다 (각 factor별로).
    """
    kappa_results = {}
    
    for factor_name, stats in factor_stats.items():
        total = stats["total_comparisons"]
        if total == 0:
            continue
        
        # Pairwise agreement 비율
        if len(stats["pairwise_agreements"]) > 0:
            p_observed = sum(stats["pairwise_agreements"]) / len(stats["pairwise_agreements"])
        else:
            p_observed = 0
        
        # Expected agreement (3개 카테고리: true, false, unsure)
        # 각 카테고리가 동일한 확률로 선택된다고 가정
        p_expected = 1.0 / 3.0
        
        if p_expected == 1.0:
            kappa = 0.0
        else:
            kappa = (p_observed - p_expected) / (1.0 - p_expected)
        
        kappa_results[factor_name] = {
            "kappa": kappa,
            "p_observed": p_observed,
            "p_expected": p_expected
        }
    
    return kappa_results


def print_results(factor_stats, kappa_results):
    """
    결과를 출력합니다.
    """
    print("=" * 90)
    print("Inter-Annotator Agreement (IAA) Analysis")
    print("(Only selected factors: age, appearance_body_status (Thin/Fat only),")
    print(" disability_status (without a disability/wheelchair only), gender, location, race_ethnicity)")
    print("=" * 90)
    print()
    
    sorted_factors = sorted(factor_stats.keys())
    
    # Factor별 상세 통계
    print("Factor-wise Agreement Statistics:")
    print("-" * 90)
    print(f"{'Factor':<30} {'Total':<10} {'Perfect':<10} {'3 Agree':<10} {'2 Agree':<10} {'No Agree':<10} {'Pairwise %':<12}")
    print("-" * 90)
    
    total_perfect = 0
    total_comparisons = 0
    total_pairwise_agreements = 0
    total_pairwise_count = 0
    
    for factor_name in sorted_factors:
        stats = factor_stats[factor_name]
        total = stats["total_comparisons"]
        
        if total == 0:
            continue
        
        total_comparisons += total
        total_perfect += stats["perfect_agreement"]
        
        pairwise_count = len(stats["pairwise_agreements"])
        pairwise_agreements = sum(stats["pairwise_agreements"])
        total_pairwise_agreements += pairwise_agreements
        total_pairwise_count += pairwise_count
        
        pairwise_pct = (pairwise_agreements / pairwise_count * 100) if pairwise_count > 0 else 0
        
        perfect_pct = (stats["perfect_agreement"] / total * 100) if total > 0 else 0
        three_pct = (stats["three_agree"] / total * 100) if total > 0 else 0
        two_pct = (stats["two_agree"] / total * 100) if total > 0 else 0
        no_pct = (stats["no_agreement"] / total * 100) if total > 0 else 0
        
        print(f"{factor_name:<30} {total:<10} {stats['perfect_agreement']:<10} ({perfect_pct:>5.1f}%) "
              f"{stats['three_agree']:<10} ({three_pct:>5.1f}%) {stats['two_agree']:<10} ({two_pct:>5.1f}%) "
              f"{stats['no_agreement']:<10} ({no_pct:>5.1f}%) {pairwise_pct:>6.2f}%")
    
    print("-" * 90)
    overall_perfect_pct = (total_perfect / total_comparisons * 100) if total_comparisons > 0 else 0
    overall_pairwise_pct = (total_pairwise_agreements / total_pairwise_count * 100) if total_pairwise_count > 0 else 0
    print(f"{'Overall':<30} {total_comparisons:<10} {total_perfect:<10} ({overall_perfect_pct:>5.1f}%) "
          f"{'':<10} {'':<10} {'':<10} {overall_pairwise_pct:>6.2f}%")
    
    print()
    print("=" * 90)
    print()
    
    # Fleiss' Kappa
    print("Fleiss' Kappa (Multi-annotator agreement):")
    print("-" * 90)
    print(f"{'Factor':<30} {'Kappa':<15} {'P(observed)':<15} {'P(expected)':<15}")
    print("-" * 90)
    
    overall_kappa = 0
    kappa_count = 0
    
    for factor_name in sorted_factors:
        if factor_name in kappa_results:
            kappa = kappa_results[factor_name]["kappa"]
            p_obs = kappa_results[factor_name]["p_observed"]
            p_exp = kappa_results[factor_name]["p_expected"]
            
            overall_kappa += kappa
            kappa_count += 1
            
            print(f"{factor_name:<30} {kappa:>6.4f}        {p_obs:>6.4f}        {p_exp:>6.4f}")
    
    if kappa_count > 0:
        avg_kappa = overall_kappa / kappa_count
        print("-" * 90)
        print(f"{'Average Kappa':<30} {avg_kappa:>6.4f}")
    
    print("=" * 90)
    print()
    
    # Kappa 해석
    print("Kappa Interpretation:")
    print("  < 0.00: Poor agreement")
    print("  0.00-0.20: Slight agreement")
    print("  0.21-0.40: Fair agreement")
    print("  0.41-0.60: Moderate agreement")
    print("  0.61-0.80: Substantial agreement")
    print("  0.81-1.00: Almost perfect agreement")
    print("=" * 90)


def main():
    """Main function."""
    script_dir = Path("/root/cs454_ct_diffusion_bias/annotations")
    
    annotation_files = [
        script_dir / "annotations_희진.json",
        script_dir / "annotations_준영.json",
        script_dir / "annotations_장현.json",
        script_dir / "annotations_승재.json"
    ]
    
    # 파일 존재 확인
    for filepath in annotation_files:
        if not filepath.exists():
            print(f"Error: {filepath} 파일을 찾을 수 없습니다.")
            return
    
    vqa_file = Path("/root/cs454_ct_diffusion_bias/evaluator/vqa_results/scores_2wise.json")
    
    # Annotation 로드
    print("Loading annotations...")
    annotations_dict = load_annotations(annotation_files)
    print(f"Loaded {len(annotations_dict)} images")
    print()
    
    # IAA 계산
    print("Calculating IAA...")
    factor_stats = calculate_agreement(annotations_dict, vqa_file)
    
    # Fleiss' Kappa 계산
    kappa_results = calculate_fleiss_kappa(factor_stats)
    
    # 결과 출력
    print_results(factor_stats, kappa_results)


if __name__ == "__main__":
    main()

