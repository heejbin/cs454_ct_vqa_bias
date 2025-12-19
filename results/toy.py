#!/usr/bin/env python3
"""
각 image_id별로 accuracy와 average_yes_probability의 평균을 계산하고,
높은 순서대로 정렬하여 full_prompt를 출력하는 스크립트
"""

import json
from collections import defaultdict

def analyze_scores(json_file_path):
    """
    JSON 파일을 읽어서 각 image_id별로 accuracy와 average_yes_probability의
    평균을 계산하고, 높은 순서대로 정렬하여 full_prompt를 출력합니다.
    """
    # JSON 파일 읽기
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # image_id별로 데이터 집계
    image_data = defaultdict(lambda: {
        'accuracies': [],
        'avg_yes_probs': [],
        'full_prompts': set()  # 중복 제거를 위해 set 사용
    })
    
    for item in data:
        image_id = item['image_id']
        summary = item['summary']
        
        image_data[image_id]['accuracies'].append(summary['accuracy'])
        image_data[image_id]['avg_yes_probs'].append(summary['average_yes_probability'])
        image_data[image_id]['full_prompts'].add(item['full_prompt'])
    
    # 각 image_id별로 평균 계산 및 결과 리스트 생성
    results = []
    for image_id, data_dict in image_data.items():
        avg_accuracy = sum(data_dict['accuracies']) / len(data_dict['accuracies'])
        avg_yes_prob = sum(data_dict['avg_yes_probs']) / len(data_dict['avg_yes_probs'])
        combined_avg = (avg_accuracy + avg_yes_prob * 100) / 2  # accuracy는 %, yes_prob는 0-1 범위
        
        # full_prompt는 하나만 선택 (보통 같은 image_id면 같은 prompt일 것)
        full_prompt = list(data_dict['full_prompts'])[0]
        
        results.append({
            'image_id': image_id,
            'avg_accuracy': avg_accuracy,
            'avg_yes_probability': avg_yes_prob,
            'combined_average': combined_avg,
            'full_prompt': full_prompt,
            'count': len(data_dict['accuracies'])
        })
    
    # combined_average 기준으로 내림차순 정렬
    results.sort(key=lambda x: x['combined_average'], reverse=True)
    
    # 결과 출력
    print("=" * 100)
    print(f"{'Rank':<6} {'Image ID':<12} {'Avg Accuracy':<15} {'Avg Yes Prob':<15} {'Combined Avg':<15} {'Count':<8}")
    print("=" * 100)
    
    for idx, result in enumerate(results, 1):
        print(f"{idx:<6} {result['image_id']:<12} "
              f"{result['avg_accuracy']:<15.2f} "
              f"{result['avg_yes_probability']:<15.4f} "
              f"{result['combined_average']:<15.2f} "
              f"{result['count']:<8}")
        print(f"\nFull Prompt: {result['full_prompt']}\n")
        print("-" * 100)
    
    return results

if __name__ == '__main__':
    json_file_path = '/root/cs454_ct_diffusion_bias/evaluator/vqa_results/scores_2wise.json'
    analyze_scores(json_file_path)

