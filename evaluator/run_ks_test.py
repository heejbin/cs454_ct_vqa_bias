import argparse
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict

def load_scores_by_attribute_value(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    data_by_attr_val = defaultdict(lambda: defaultdict(list))
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Parsing {len(data)} images from {filepath}...")

        for item in data:
            assigned_attrs = item.get('assigned_attributes', {})
            vqa_results = item.get('vqa_results', {})

            if not assigned_attrs or not vqa_results:
                continue

            for attr_name, result in vqa_results.items():
                if attr_name not in assigned_attrs:
                    continue
                
                attr_value = assigned_attrs[attr_name]
                
                if isinstance(result, dict) and 'yes_probability' in result:
                    score = result['yes_probability']
                    data_by_attr_val[attr_name][attr_value].append(score)

        return data_by_attr_val

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def plot_cdf_comparison(scores1, scores2, title, filename, label1, label2):
    def ecdf(data):
        n = len(data)
        x = np.sort(data)
        y = np.arange(1, n + 1) / n
        return x, y

    plt.figure(figsize=(8, 6))
    
    x1, y1 = ecdf(scores1)
    x2, y2 = ecdf(scores2)

    plt.plot(x1, y1, marker='.', linestyle='none', label=label1, alpha=0.5, markersize=3)
    plt.plot(x2, y2, marker='.', linestyle='none', label=label2, alpha=0.5, markersize=3)
    
    plt.title(f"ECDF Comparison: {title}")
    plt.xlabel('Yes Probability')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Perform KS test on VQA scores by attribute value.")
    parser.add_argument("file1", type=str, help="Path to the first VQA scores JSON file")
    parser.add_argument("file2", type=str, help="Path to the second VQA scores JSON file")
    parser.add_argument("--label1", type=str, help="Label for the first file (default: filename)")
    parser.add_argument("--label2", type=str, help="Label for the second file (default: filename)")
    parser.add_argument("--output_dir", type=str, default="ks_results", help="Root directory to save results")
    
    args = parser.parse_args()
    
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    label1 = args.label1 if args.label1 else os.path.splitext(os.path.basename(args.file1))[0]
    label2 = args.label2 if args.label2 else os.path.splitext(os.path.basename(args.file2))[0]

    print(f"--- Loading Data ---")
    data1 = load_scores_by_attribute_value(args.file1)
    data2 = load_scores_by_attribute_value(args.file2)

    if not data1 or not data2:
        print("Failed to load data.")
        return

    print(f"\n--- Performing KS Tests: {label1} vs {label2} ---")
    
    results = []

    for attr_name, vals1 in data1.items():
        if attr_name not in data2:
            continue
        
        vals2 = data2[attr_name]
        
        # Aggregation lists for Theme-level test
        theme_scores1 = []
        theme_scores2 = []

        for val_name, scores1 in vals1.items():
            if val_name not in vals2:
                continue

            scores2 = vals2[val_name]
            
            # Collect scores for theme level comparison
            theme_scores1.extend(scores1)
            theme_scores2.extend(scores2)
            
            if len(scores1) < 10 or len(scores2) < 10:
                 continue

            ks_stat, p_val = stats.ks_2samp(scores1, scores2)
            
            results.append({
                'attribute': attr_name,
                'value': val_name,
                'ks_stat': ks_stat,
                'p_value': p_val,
                'count1': len(scores1),
                'count2': len(scores2),
                'scores1': scores1,
                'scores2': scores2,
                'type': 'value'
            })
            
        # Perform Theme-level KS Test
        if theme_scores1 and theme_scores2:
            ks_stat, p_val = stats.ks_2samp(theme_scores1, theme_scores2)
            results.append({
                'attribute': attr_name,
                'value': '(TOTAL)',
                'ks_stat': ks_stat,
                'p_value': p_val,
                'count1': len(theme_scores1),
                'count2': len(theme_scores2),
                'scores1': theme_scores1,
                'scores2': theme_scores2,
                'type': 'theme'
            })

    # Sort: First by Type (Theme first), then by KS Statistic
    results.sort(key=lambda x: x['ks_stat'], reverse=True)

    # Print to console and generate plots
    l1_head = (label1[:6])
    l2_head = (label2[:6])
    
    print(f"\n{'Attribute':<20} | {'Value':<20} | {'KS Stat':<10} | {'P-value':<12} | {l1_head:<8} | {l2_head:<8}")
    print("-" * 94)

    # Prepare CSV data
    csv_rows = [['Attribute', 'Value', 'KS_Statistic', 'P_Value', f'{label1}_Count', f'{label2}_Count', 'Type']]

    for i, res in enumerate(results):
        # Console Output
        attr = res['attribute']
        val = str(res['value'])[:20] 
        ks = f"{res['ks_stat']:.4f}"
        p = f"{res['p_value'] * 100:.2f}%"
        c1 = res['count1']
        c2 = res['count2']
        
        row_str = f"{attr:<20} | {val:<20} | {ks:<10} | {p:<12} | {c1:<8} | {c2:<8}"
        if res['type'] == 'theme':
            print(f"\033[1m{row_str}\033[0m")
        else:
            print(row_str)

        # CSV Data
        csv_rows.append([res['attribute'], res['value'], res['ks_stat'], res['p_value'], res['count1'], res['count2'], res['type']])

        # Plot Generation (All)
        rank = i + 1
        title = f"Rank #{rank} (KS: {res['ks_stat']:.4f})\n{res['attribute']} - {res['value']}"
        
        # Sanitize names for file paths
        safe_attr = res['attribute'].replace(" ", "_").replace("/", "_").replace(":", "")
        safe_val = str(res['value']).replace(" ", "_").replace("/", "_").replace(":", "")
        
        # Create attribute directory
        attr_dir = os.path.join(plots_dir, safe_attr)
        os.makedirs(attr_dir, exist_ok=True)
        
        filename = os.path.join(attr_dir, f"{safe_val}.png")
        plot_cdf_comparison(res['scores1'], res['scores2'], title, filename, label1, label2)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "ks_summary.csv")
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\nSuccessfully saved summary table to: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    print(f"Successfully saved {len(results)} plots to: {plots_dir}")

if __name__ == "__main__":
    main()
