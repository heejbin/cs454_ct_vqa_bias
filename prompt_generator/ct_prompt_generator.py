#!/usr/bin/env python3
import copy
import jsonc
import itertools
import random
import sys
import os

from typing import List, Dict, Tuple, Set, Optional

class CTPromptGenerator:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = jsonc.load(f)
        
        cartesian_attributes = self.config.get('cartesian_attributes', [])
        split_attributes = self.config.get('split_attributes', [])
        split_attribute_labels = self.config.get('split_attribute_labels', [])
        merged_attributes = self.config.get('merged_attributes', [])
        merged_attribute_labels = self.config.get('merged_attribute_labels', [])

        self.factors = self.config['factors']
        self.cartesian_factors = [f for f in self.factors if f['attribute'] in cartesian_attributes]
        self.t = self.config['t']
        self.template = self.config['base_prompt_template']

        self.cartesian_factor_labels = [f['labels'] for f in self.factors if f['attribute'] in cartesian_attributes]
        self.non_cartesian_factor_names = [f['name'] for f in self.factors if f['attribute'] not in cartesian_attributes]
        self.non_cartesian_factor_labels = [f['labels'] for f in self.factors if f['attribute'] not in cartesian_attributes]

        self.split_attributes = {
            tuple(split_attributes[i]): split_attribute_labels[i]
            for i in range(len(split_attributes))
        }
        self.merged_attributes = {
            merged_attributes[i]: merged_attribute_labels[i]
            for i in range(len(merged_attributes))
        }
        self.split_merged_attribute_map = {
            tuple(split_attributes[i]): merged_attributes[i]
            for i in range(len(split_attributes))
        }

        print(f"CT Prompt Generator: {self.t}-wise Coverage")
        print(f"Factors: {len(self.factors)}")
        for i, factor in enumerate(self.factors):
            print(f"  {i+1}. {factor['name']}: {len(factor['labels'])} labels")
        print(f"\nTotal combinations (Full Cartesian): {self._get_full_size():,}")
        print(f"{'='*80}\n")
    
    def _get_full_size(self) -> int:
        size = 1
        for labels in self.non_cartesian_factor_labels:
            size *= len(labels)
        for labels in self.cartesian_factor_labels:
            size *= len(labels)
        return size
    
    def _load_previous_prompts(self, previous_t: int, output_path_hint: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """
        Load test cases from previous t-1-wise result file.
        Returns None if file doesn't exist or t=2 (no previous file).
        
        Args:
            previous_t: The t value of the previous results to load
            output_path_hint: Optional hint for where to look for the previous file
        """
        if previous_t < 2:
            return None
        
        possible_paths = []
        
        # If output_path_hint is provided, try to derive previous file path from it
        if output_path_hint:
            output_dir = os.path.dirname(output_path_hint) if os.path.dirname(output_path_hint) else '.'
            output_basename = os.path.basename(output_path_hint)
            # Replace current t with previous_t in filename
            prev_basename = output_basename.replace(f"{self.t}wise", f"{previous_t}wise")
            possible_paths.append(os.path.join(output_dir, prev_basename))
        
        # Try to find previous prompt file in common locations
        # Look for ct_prompts_{previous_t}wise.json or ct_prompts_{previous_t}wise.jsonc
        possible_paths.extend([
            f"prompt/ct_prompts_{previous_t}wise.json",
            f"prompt/ct_prompts_{previous_t}wise.jsonc",
            f"ct_prompts_{previous_t}wise.json",
            f"ct_prompts_{previous_t}wise.jsonc",
        ])
        
        # Also try relative to config file directory if available
        if hasattr(self, 'config_path'):
            config_dir = os.path.dirname(self.config_path) if os.path.dirname(self.config_path) else '.'
            possible_paths.extend([
                os.path.join(config_dir, f"../prompt/ct_prompts_{previous_t}wise.json"),
                os.path.join(config_dir, f"../prompt/ct_prompts_{previous_t}wise.jsonc"),
                os.path.join(config_dir, f"ct_prompts_{previous_t}wise.json"),
                os.path.join(config_dir, f"ct_prompts_{previous_t}wise.jsonc"),
            ])
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                print(f"Loading previous {previous_t}-wise results from: {abs_path}")
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        prev_data = jsonc.load(f)
                    
                    # Extract test cases (remove cartesian factors if present)
                    prev_test_cases = []
                    cartesian_factor_names = [f['name'] for f in self.cartesian_factors]
                    
                    for tc in prev_data.get('test_cases', []):
                        assignment = tc.get('assignment', {})
                        # Remove cartesian factors to get base test cases
                        base_assignment = {k: v for k, v in assignment.items() 
                                         if k not in cartesian_factor_names}
                        if base_assignment:
                            prev_test_cases.append(base_assignment)
                    
                    # Remove duplicates
                    seen = set()
                    unique_test_cases = []
                    for tc in prev_test_cases:
                        tc_tuple = tuple(sorted(tc.items()))
                        if tc_tuple not in seen:
                            seen.add(tc_tuple)
                            unique_test_cases.append(tc)
                    
                    print(f"Loaded {len(unique_test_cases)} unique test cases from previous {previous_t}-wise")
                    return unique_test_cases
                except Exception as e:
                    print(f"Warning: Could not load previous results from {abs_path}: {e}")
                    continue
        
        print(f"No previous {previous_t}-wise results found, starting fresh")
        return None
    
    def _get_uncovered_t_tuples(self, t: int, existing_test_cases: List[Dict[str, str]]) -> Set[Tuple]:
        """
        Get all uncovered t-tuples, excluding those already covered by existing test cases.
        """
        # Get all possible t-tuples
        all_tuples = self._get_all_t_tuples(t)
        
        if not existing_test_cases:
            return all_tuples
        
        # Remove tuples already covered by existing test cases
        covered_tuples = set()
        for test_case in existing_test_cases:
            # Convert test_case dict to list of values in factor order
            assignment = [test_case.get(factor_name, None) 
                         for factor_name in self.non_cartesian_factor_names]
            
            # Check all t-tuples that can be formed from this test case
            for factor_indices in itertools.combinations(range(len(assignment)), t):
                if all(assignment[i] is not None for i in factor_indices):
                    label_combo = tuple(assignment[i] for i in factor_indices)
                    covered_tuples.add((factor_indices, label_combo))
        
        uncovered = all_tuples - covered_tuples
        return uncovered
    
    def _generate_greedy_covering(self, t: int, existing_test_cases: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Generate test cases using greedy covering algorithm.
        If existing_test_cases is provided, only generate new test cases for uncovered t-tuples.
        """
        if existing_test_cases is None:
            existing_test_cases = []
        
        # Get uncovered t-tuples (excluding those already covered)
        uncovered = self._get_uncovered_t_tuples(t, existing_test_cases)
        initial_size = len(uncovered)
        
        print(f"Uncovered {t}-tuples: {len(uncovered):,} (out of {self._get_all_t_tuples_count(t):,} total)")
        
        # Start with existing test cases
        test_cases = copy.deepcopy(existing_test_cases)
        cartesian_test_cases = []

        iteration = 0
        max_iterations = 10000
        
        while uncovered and iteration < max_iterations:
            best_assignment = self._find_best_assignment(uncovered, t)
            
            if len(best_assignment) == 0:
                best_assignment = [random.choice(labels) for labels in self.non_cartesian_factor_labels]
            
            assignment_dict = {}
            for i, factor_name in enumerate(self.non_cartesian_factor_names):
                assignment_dict[factor_name] = best_assignment[i]
            
            test_cases.append(assignment_dict)
            
            uncovered = self._remove_covered(uncovered, best_assignment, t)
            
            iteration += 1

            if iteration % max(1, max_iterations // 10) == 0:
                covered_pct = (1 - len(uncovered) / initial_size) * 100 if initial_size > 0 else 100
                print(f"Progress: {iteration}/{max_iterations}, Coverage: {covered_pct:.1f}%")

        if uncovered:
            print(f"Maximum iterations reached. Uncovered combinations: {len(uncovered)}")
        
        new_test_cases = len(test_cases) - len(existing_test_cases)
        if new_test_cases > 0:
            print(f"Added {new_test_cases} new test cases for {t}-wise coverage")

        if len(self.cartesian_factor_labels) > 0:
            cartesian_combinations = list(itertools.product(*[f['labels'] for f in self.cartesian_factors]))

            for test_case in test_cases:
                for combo in cartesian_combinations:
                    new_test_case = test_case.copy()
                    for i, cartesian_factor in enumerate(self.cartesian_factors):
                        new_test_case[cartesian_factor['name']] = combo[i]
                    cartesian_test_cases.append(new_test_case)

            print(f"{t}-wise: {len(cartesian_test_cases)} total test cases ({len(existing_test_cases) * len(cartesian_combinations)} from previous, {new_test_cases * len(cartesian_combinations)} new)\n")
            return cartesian_test_cases
        else:
            print(f"{t}-wise: {len(test_cases)} total test cases ({len(existing_test_cases)} from previous, {new_test_cases} new)\n")
            return test_cases
    
    def _get_all_t_tuples_count(self, t: int) -> int:
        """Get the total number of possible t-tuples (for reporting)."""
        count = 0
        for factor_indices in itertools.combinations(range(len(self.non_cartesian_factor_names)), t):
            size = 1
            for i in factor_indices:
                size *= len(self.non_cartesian_factor_labels[i])
            count += size
        return count
    
    def _get_all_t_tuples(self, t: int) -> Set[Tuple]:
        uncovered = set()
        
        for factor_indices in itertools.combinations(range(len(self.factors) - len(self.cartesian_factors)), t):
            selected_labels = [self.non_cartesian_factor_labels[i] for i in factor_indices]
            
            for label_combo in itertools.product(*selected_labels):
                uncovered.add((factor_indices, label_combo))
        
        return uncovered
    
    def _find_best_assignment(self, uncovered: Set[Tuple], t: int) -> List[str]:
        best_assignment = []
        best_coverage = 0
        
        sample_size = min(1000, self._get_full_size())
        
        for _ in range(sample_size):
            assignment = [random.choice(labels) for labels in self.non_cartesian_factor_labels]
            
            coverage = self._count_coverage(assignment, uncovered, t)
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_assignment = assignment
        
        return best_assignment
    
    def _count_coverage(self, assignment: List[str], uncovered: Set[Tuple], t: int) -> int:
        count = 0
        
        # 현재 assignment로부터 모든 t-tuple 생성하고 uncovered에서 O(1) lookup
        for factor_indices in itertools.combinations(range(len(assignment)), t):
            label_combo = tuple(assignment[i] for i in factor_indices)
            
            if (factor_indices, label_combo) in uncovered:
                count += 1
        
        return count
    
    def _remove_covered(self, uncovered: Set[Tuple], assignment: List[str], t: int) -> Set[Tuple]:
        # 현재 assignment로 커버되는 tuple들을 생성하고 제거
        covered_tuples = set()
        for factor_indices in itertools.combinations(range(len(assignment)), t):
            label_combo = tuple(assignment[i] for i in factor_indices)
            covered_tuples.add((factor_indices, label_combo))
        
        # uncovered에서 covered_tuples 제거
        return uncovered - covered_tuples
    
    def generate_prompts(self, output_path_hint: Optional[str] = None) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Generate prompts incrementally:
        - t=2: Generate all 2-wise combinations
        - t>2: Load previous t-1-wise results and add new test cases for uncovered t-tuples
        
        Args:
            output_path_hint: Optional hint for where to look for previous results
        """
        previous_t = self.t - 1
        existing_test_cases = None
        
        if self.t > 2:
            # Try to load previous t-1-wise results
            existing_test_cases = self._load_previous_prompts(previous_t, output_path_hint=output_path_hint)
        
        # Generate test cases (new ones if t=2, or additional ones if t>2)
        test_cases = self._generate_greedy_covering(self.t, existing_test_cases=existing_test_cases)
        
        prompts = []
        for assignment in test_cases:
            merged_assignment = copy.copy(assignment)

            for split_attributes, labels in self.split_attributes.items():
                if all(attribute in assignment for attribute in split_attributes):
                    split_attribute_label = [assignment[attribute] for attribute in split_attributes]
                    split_attribute_index = -1

                    for label_index in range(len(labels)):
                        if labels[label_index] == split_attribute_label:
                            split_attribute_index = label_index
                            break

                    merged_attribute_name = self.split_merged_attribute_map[split_attributes]

                    for attribute in split_attributes:
                        del merged_assignment[attribute]

                    merged_assignment[merged_attribute_name] = self.merged_attributes[merged_attribute_name][split_attribute_index]
            try:
                prompt = self.template.format(**merged_assignment)
                prompts.append(prompt)
            except KeyError as e:
                raise ValueError(f"Template error: {e}")
        
        return prompts, test_cases
    
    def save_prompt_set(self, output_path: str):
        prompts, assignments = self.generate_prompts(output_path_hint=output_path)
        
        output = {
            "t": self.t,
            "base_prompt_template": self.template,
            "factors": self.factors,
            "test_cases": []
        }
        
        for idx, (prompt, assignment) in enumerate(zip(prompts, assignments)):
            # aliased_assignment = copy.copy(assignment)
            #
            # for name, factor in assignment.items():
            #     if name in self.alias_factors:
            #         del aliased_assignment[name]
            #
            #         for i in range(len(self.alias_factors[name]['labels'])):
            #             aliased_factor = self.alias_factors[name]['labels'][i]
            #             saved_name = self.alias_factors[name]['save_names']
            #             saved_label = self.alias_factors[name]['save_labels'][i]
            #
            #             if factor == aliased_factor:
            #                 for aliased_name, aliased_label in zip(saved_name, saved_label):
            #                     aliased_assignment[aliased_name] = aliased_label
            #                 break

            output["test_cases"].append({
                "id": idx,
                "assignment": assignment,
                "prompt": prompt
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            jsonc.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Total prompts: {len(prompts)}")
        print(f"Full Cartesian: {self._get_full_size()}")
        
        return output

def main():
    if len(sys.argv) < 2:
        print("Usage: python ct_prompt_generator.py <config_json_path> [output_json_path]")
        print("\nExample:")
        print("python ct_prompt_generator.py prompt_config/ct_config_2wise.jsonc")
        print("python ct_prompt_generator.py prompt_config/ct_config_3wise.jsonc")
        print("\nNote: For t>2, the script will automatically load previous t-1-wise results")
        print("      (e.g., ct_prompts_2wise.json for 3-wise, ct_prompts_3wise.json for 4-wise)")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        # Generate output path from config path
        # ct_config_2wise.jsonc → prompt/ct_prompts_2wise.json
        config_dir = os.path.dirname(config_path) if os.path.dirname(config_path) else '.'
        config_basename = os.path.basename(config_path)
        
        # Extract t value and create output filename
        if 'config' in config_basename:
            output_basename = config_basename.replace('config', 'prompts').replace('.jsonc', '.json').replace('.json', '.json')
            # Ensure .json extension
            if not output_basename.endswith('.json'):
                output_basename = output_basename.rsplit('.', 1)[0] + '.json'
        else:
            # Fallback: just change extension
            output_basename = os.path.splitext(config_basename)[0] + '_prompts.json'
        
        # Try to put in prompt/ directory if it exists, otherwise same directory
        prompt_dir = os.path.join(config_dir, 'prompt')
        if os.path.exists(prompt_dir):
            output_path = os.path.join(prompt_dir, output_basename)
        else:
            # Try parent directory
            parent_prompt_dir = os.path.join(os.path.dirname(config_dir), 'prompt')
            if os.path.exists(parent_prompt_dir):
                output_path = os.path.join(parent_prompt_dir, output_basename)
            else:
                # Use same directory as config
                output_path = os.path.join(config_dir, output_basename)
    
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    generator = CTPromptGenerator(config_path)
    generator.save_prompt_set(output_path)

if __name__ == "__main__":
    main()