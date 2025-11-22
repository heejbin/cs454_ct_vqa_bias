#!/usr/bin/env python3
import copy
import jsonc
import itertools
import random
import sys

from typing import List, Dict, Tuple, Set

class CTPromptGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = jsonc.load(f)
        
        cartesian_attributes = self.config.get('cartesian_attributes', [])

        self.factors = self.config['factors']
        self.cartesian_factors = [f for f in self.factors if f['attribute'] in cartesian_attributes]
        self.t = self.config['t']
        self.template = self.config['base_prompt_template']

        self.cartesian_factor_labels = [f['labels'] for f in self.factors if f['attribute'] in cartesian_attributes]
        self.non_cartesian_factor_names = [f['name'] for f in self.factors if f['attribute'] not in cartesian_attributes]
        self.non_cartesian_factor_labels = [f['labels'] for f in self.factors if f['attribute'] not in cartesian_attributes]

        self.alias_factors = {}
        for alias_factor in [f for f in self.factors if 'save_names' in f]:
            self.alias_factors[alias_factor['name']] = alias_factor

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
    
    def _generate_greedy_covering(self, t: int) -> List[Dict[str, str]]:
        
        uncovered = self._get_all_t_tuples(t)
        initial_size = len(uncovered)
        
        test_cases = []
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
                covered_pct = (1 - len(uncovered) / initial_size) * 100
                print(f"Progress: {iteration}/{max_iterations}, Coverage: {covered_pct:.1f}%")

        if uncovered:
            print(f"Maximum iterations reached. Uncovered combinations: {len(uncovered)}")

        if len(self.cartesian_factor_labels) > 0:
            cartesian_combinations = list(itertools.product(*[f['labels'] for f in self.cartesian_factors]))

            for test_case in test_cases:
                for combo in cartesian_combinations:
                    new_test_case = test_case.copy()
                    for i, cartesian_factor in enumerate(self.cartesian_factors):
                        new_test_case[cartesian_factor['name']] = combo[i]
                    cartesian_test_cases.append(new_test_case)

            print(f"{t}-wise: {len(cartesian_test_cases)} test cases generated\n")
            return cartesian_test_cases
        else:
            print(f"{t}-wise: {len(test_cases)} test cases generated\n")
            return test_cases
    
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
    
    def generate_prompts(self) -> Tuple[List[str], List[Dict[str, str]]]:
        test_cases = self._generate_greedy_covering(self.t)
        prompts = []
        for assignment in test_cases:
            try:
                prompt = self.template.format(**assignment)
                prompts.append(prompt)
            except KeyError as e:
                raise ValueError(f"Template error: {e}")
        
        return prompts, test_cases
    
    def save_prompt_set(self, output_path: str):
        prompts, assignments = self.generate_prompts()
        
        output = {
            "t": self.t,
            "base_prompt_template": self.template,
            "factors": self.factors,
            "test_cases": []
        }
        
        for idx, (prompt, assignment) in enumerate(zip(prompts, assignments)):
            aliased_assignment = copy.copy(assignment)

            for name, factor in assignment.items():
                if name in self.alias_factors:
                    del aliased_assignment[name]

                    for i in range(len(self.alias_factors[name]['labels'])):
                        aliased_factor = self.alias_factors[name]['labels'][i]
                        saved_name = self.alias_factors[name]['save_names']
                        saved_label = self.alias_factors[name]['save_labels'][i]

                        if factor == aliased_factor:
                            for aliased_name, aliased_label in zip(saved_name, saved_label):
                                aliased_assignment[aliased_name] = aliased_label
                            break

            output["test_cases"].append({
                "id": idx,
                "assignment": aliased_assignment,
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
        print("python ct_prompt_generator.py /path/to/ct_config_2wise.json /path/to/ct_prompts_2wise.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        # ct_config_2wise.json → ct_prompts_2wise.json
        base_name = config_path.replace('config', 'prompts')
        output_path = base_name
    
    generator = CTPromptGenerator(config_path)
    generator.save_prompt_set(output_path)

if __name__ == "__main__":
    main()