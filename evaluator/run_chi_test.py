import argparse
import json
import numpy as np
from scipy import stats
import os
import sys
from collections import defaultdict, Counter

def load_prompt_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_chi_squared(data):
    factors = data.get('factors', [])
    test_cases = data.get('test_cases', [])
    
    if not factors or not test_cases:
        print("Error: Invalid data format. Missing 'factors' or 'test_cases'.")
        return []

    print(f"Loaded {len(test_cases)} prompts.")
    
    results = []

    # Map factor names to their defined labels
    factor_dict = {f['name']: f['labels'] for f in factors}

    for factor_name, defined_labels in factor_dict.items():
        # Count observed occurrences for this factor
        observed_counts = Counter()
        for tc in test_cases:
            assignment = tc.get('assignment', {})
            val = assignment.get(factor_name)
            if val:
                observed_counts[val] += 1
        
        # Prepare observed and expected arrays
        # We must ensure we count 0 for labels that appear 0 times but are defined
        obs = []
        
        # Expected count for uniform distribution
        total_count = len(test_cases)
        num_labels = len(defined_labels)
        expected_val = total_count / num_labels
        exp = [expected_val] * num_labels
        
        # Fill observed based on the order of defined_labels
        for label in defined_labels:
            obs.append(observed_counts[label])
            
        # Perform Chi-squared test
        # ddof=0 is standard for this test (degrees of freedom = k - 1)
        chi2_stat, p_val = stats.chisquare(f_obs=obs, f_exp=exp)

        results.append({
            'factor': factor_name,
            'num_labels': num_labels,
            'chi2_stat': chi2_stat,
            'p_value': p_val,
            'observed': dict(zip(defined_labels, obs)),
            'expected': expected_val
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Perform Chi-squared test on generated prompts to check for uniform distribution of attributes.")
    parser.add_argument("prompt_file", type=str, help="Path to the generated prompts JSON file (e.g., ct_prompts_2wise.json)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for the test (default: 0.05)")
    
    args = parser.parse_args()
    
    print(f"--- Loading Prompt Data from {args.prompt_file} ---")
    data = load_prompt_data(args.prompt_file)
    
    if not data:
        return

    print(f"\n--- Performing Chi-squared Tests (Uniformity Check) ---")
    results = analyze_chi_squared(data)
    
    if not results:
        print("No results generated.")
        return

    # Sort results by p-value (ascending - most significant deviations first)
    results.sort(key=lambda x: x['p_value'])

    print(f"\n{'Factor':<30} | {'Labels':<6} | {'Chi2 Stat':<10} | {'P-value':<12} | {'Result':<10}")
    print("-" * 80)
    
    for res in results:
        factor = res['factor']
        n_labels = res['num_labels']
        chi2 = f"{res['chi2_stat']:.4f}"
        p_val_num = res['p_value']
        p_val = f"{p_val_num:.6f}"
        
        is_significant = p_val_num < args.alpha
        result_str = "REJECT" if is_significant else "ACCEPT"
        color_code = "\033[91m" if is_significant else "\033[92m" # Red for Reject (Deviation), Green for Accept (Uniform) 
        
        print(f"{factor:<30} | {n_labels:<6} | {chi2:<10} | {p_val:<12} | {color_code}{result_str}\033[0m")
        
        # Optional: Print details if rejected or verbose
        if is_significant:
            print(f"  > Expected per label: {res['expected']:.2f}")
            print(f"  > Observed: {res['observed']}")
            print("")

    print(f"\nInterpretation:")
    print(f" - ACCEPT (p > {args.alpha}): No significant evidence to reject the hypothesis that labels are uniformly distributed.")
    print(f" - REJECT (p < {args.alpha}): Statistically significant deviation from uniform distribution.")

if __name__ == "__main__":
    main()
