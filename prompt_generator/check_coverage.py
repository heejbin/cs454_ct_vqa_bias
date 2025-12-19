#!/usr/bin/env python3
"""
Check if all t-wise combinations are covered in the generated CT prompts.
"""
import json
import jsonc
import itertools
import sys
from typing import List, Dict, Set, Tuple

def load_prompt_file(file_path: str) -> Dict:
    """Load prompt JSON file (supports both .json and .jsonc)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonc'):
            return jsonc.load(f)
        else:
            return json.load(f)

def get_non_cartesian_factors(data: Dict) -> Tuple[List[str], List[List[str]]]:
    """
    Extract non-cartesian factors and their labels.
    Returns (factor_names, factor_labels_list).
    """
    factors = data.get('factors', [])
    cartesian_attributes = data.get('cartesian_attributes', [])
    
    factor_names = []
    factor_labels = []
    
    for factor in factors:
        attribute = factor.get('attribute', '')
        if attribute not in cartesian_attributes:
            factor_names.append(factor['name'])
            factor_labels.append(factor['labels'])
    
    return factor_names, factor_labels

def get_all_t_tuples(factor_names: List[str], factor_labels: List[List[str]], t: int) -> Set[Tuple]:
    """
    Generate all possible t-tuples.
    Returns set of (factor_indices, label_combo) tuples.
    """
    all_tuples = set()
    
    num_factors = len(factor_names)
    for factor_indices in itertools.combinations(range(num_factors), t):
        selected_labels = [factor_labels[i] for i in factor_indices]
        
        for label_combo in itertools.product(*selected_labels):
            all_tuples.add((factor_indices, label_combo))
    
    return all_tuples

def get_covered_tuples(test_case: Dict[str, str], factor_names: List[str], t: int) -> Set[Tuple]:
    """
    Extract all t-tuples covered by a test case.
    Returns set of (factor_indices, label_combo) tuples.
    """
    covered = set()
    
    # Convert test_case dict to list of values in factor order
    assignment = []
    for factor_name in factor_names:
        if factor_name in test_case:
            assignment.append(test_case[factor_name])
        else:
            assignment.append(None)
    
    # Generate all t-tuples from this assignment
    for factor_indices in itertools.combinations(range(len(assignment)), t):
        if all(assignment[i] is not None for i in factor_indices):
            label_combo = tuple(assignment[i] for i in factor_indices)
            covered.add((factor_indices, label_combo))
    
    return covered

def check_coverage(file_path: str, verbose: bool = False) -> bool:
    """
    Check if all t-wise combinations are covered.
    Returns True if all covered, False otherwise.
    """
    print(f"Loading prompt file: {file_path}")
    data = load_prompt_file(file_path)
    
    t = data.get('t', 2)
    test_cases = data.get('test_cases', [])
    
    print(f"\nChecking {t}-wise coverage")
    print(f"Total test cases: {len(test_cases)}")
    
    # Get non-cartesian factors
    factor_names, factor_labels = get_non_cartesian_factors(data)
    print(f"Non-cartesian factors: {len(factor_names)}")
    for i, name in enumerate(factor_names):
        print(f"  {i+1}. {name}: {len(factor_labels[i])} labels")
    
    # Generate all possible t-tuples
    print(f"\nGenerating all possible {t}-tuples...")
    all_tuples = get_all_t_tuples(factor_names, factor_labels, t)
    total_tuples = len(all_tuples)
    print(f"Total possible {t}-tuples: {total_tuples:,}")
    
    # Collect all covered tuples from test cases
    print(f"\nChecking coverage from test cases...")
    all_covered = set()
    
    for idx, test_case_data in enumerate(test_cases):
        assignment = test_case_data.get('assignment', {})
        covered = get_covered_tuples(assignment, factor_names, t)
        all_covered.update(covered)
        
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_cases)} test cases, "
                  f"covered {len(all_covered):,}/{total_tuples:,} tuples")
    
    # Check coverage
    uncovered = all_tuples - all_covered
    coverage_pct = (len(all_covered) / total_tuples * 100) if total_tuples > 0 else 100
    
    print(f"\n{'='*80}")
    print(f"Coverage Results:")
    print(f"  Total {t}-tuples: {total_tuples:,}")
    print(f"  Covered: {len(all_covered):,}")
    print(f"  Uncovered: {len(uncovered):,}")
    print(f"  Coverage: {coverage_pct:.2f}%")
    print(f"{'='*80}")
    
    if uncovered:
        print(f"\n⚠️  WARNING: {len(uncovered)} {t}-tuples are NOT covered!")
        
        if verbose and len(uncovered) <= 100:
            print("\nUncovered tuples (first 100):")
            for i, (factor_indices, label_combo) in enumerate(list(uncovered)[:100]):
                factor_names_str = [factor_names[idx] for idx in factor_indices]
                print(f"  {i+1}. {dict(zip(factor_names_str, label_combo))}")
        elif verbose:
            print("\nUncovered tuples (first 20):")
            for i, (factor_indices, label_combo) in enumerate(list(uncovered)[:20]):
                factor_names_str = [factor_names[idx] for idx in factor_indices]
                print(f"  {i+1}. {dict(zip(factor_names_str, label_combo))}")
            print(f"  ... and {len(uncovered) - 20} more")
        
        return False
    else:
        print(f"\n✅ SUCCESS: All {t}-tuples are covered!")
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_coverage.py <prompt_json_path> [--verbose]")
        print("\nExample:")
        print("  python check_coverage.py prompt/ct_prompts_2wise.json")
        print("  python check_coverage.py prompt/ct_prompts_3wise.json --verbose")
        sys.exit(1)
    
    file_path = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    try:
        is_covered = check_coverage(file_path, verbose=verbose)
        sys.exit(0 if is_covered else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

