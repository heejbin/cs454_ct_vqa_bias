#!/usr/bin/env python3
import json
import itertools
import random
import sys

from typing import List, Dict, Tuple, Set

class CTPromptGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.factors = self.config['factors']
        self.t = self.config['t']
        self.template = self.config['base_prompt_template']
        self.factor_names = [f['name'] for f in self.factors]
        self.factor_labels = [f['labels'] for f in self.factors]
        
        print(f"CT Prompt Generator: {self.t}-wise Coverage")
        print(f"Factors: {len(self.factors)}")
        for i, factor in enumerate(self.factors):
            print(f"  {i+1}. {factor['name']}: {len(factor['labels'])} labels")
        print(f"\nTotal combinations (Full Cartesian): {self._get_full_size():,}")
        print(f"{'='*80}\n")
    
    def _get_full_size(self) -> int:
        size = 1
        for labels in self.factor_labels:
            size *= len(labels)
        return size
    
    def generate_2wise(self) -> List[Dict[str, str]]:
        return self._generate_greedy_covering(2)
    
    def _generate_greedy_covering(self, t: int) -> List[Dict[str, str]]:
        
        uncovered = self._get_all_t_tuples(t)
        initial_size = len(uncovered)
        
        test_cases = []
        iteration = 0
        max_iterations = 10000
        
        while uncovered and iteration < max_iterations:
            best_assignment = self._find_best_assignment(uncovered, t)
            
            if best_assignment is None:
                best_assignment = [random.choice(labels) for labels in self.factor_labels]
            
            assignment_dict = {}
            for i, factor_name in enumerate(self.factor_names):
                assignment_dict[factor_name] = best_assignment[i]
            
            test_cases.append(assignment_dict)
            
            uncovered = self._remove_covered(uncovered, best_assignment, t)
            
            iteration += 1

            if iteration % max(1, max_iterations // 10) == 0:
                covered_pct = (1 - len(uncovered) / initial_size) * 100
                print(f"Progress: {iteration}/{max_iterations}, Coverage: {covered_pct:.1f}%")
        
        if uncovered:
            print(f"Maximum iterations reached. Uncovered combinations: {len(uncovered)}")
        
        print(f"{t}-wise: {len(test_cases)} test cases generated\n")
        return test_cases
    
    def _get_all_t_tuples(self, t: int) -> Set[Tuple]:
        uncovered = set()
        
        for factor_indices in itertools.combinations(range(len(self.factors)), t):
            selected_labels = [self.factor_labels[i] for i in factor_indices]
            
            for label_combo in itertools.product(*selected_labels):
                uncovered.add((factor_indices, label_combo))
        
        return uncovered
    
    def _find_best_assignment(self, uncovered: Set[Tuple], t: int) -> List[str]:
        best_assignment = None
        best_coverage = 0
        
        sample_size = min(1000, self._get_full_size())
        
        for _ in range(sample_size):
            assignment = [random.choice(labels) for labels in self.factor_labels]
            
            coverage = self._count_coverage(assignment, uncovered, t)
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_assignment = assignment
        
        return best_assignment
    
    def _count_coverage(self, assignment: List[str], uncovered: Set[Tuple], t: int) -> int:
        count = 0
        
        for factor_indices, label_combo in uncovered:
            match = True
            for i, factor_idx in enumerate(factor_indices):
                if assignment[factor_idx] != label_combo[i]:
                    match = False
                    break
            
            if match:
                count += 1
        
        return count
    
    def _remove_covered(self, uncovered: Set[Tuple], assignment: List[str], t: int) -> Set[Tuple]:
        new_uncovered = set()
        
        for factor_indices, label_combo in uncovered:
            match = True
            for i, factor_idx in enumerate(factor_indices):
                if assignment[factor_idx] != label_combo[i]:
                    match = False
                    break
            
            if not match:
                new_uncovered.add((factor_indices, label_combo))
        
        return new_uncovered
    
    def generate_prompts(self) -> Tuple[List[str], List[Dict[str, str]]]:
        if self.t == 2:
            test_cases = self.generate_2wise()
        else:
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
            output["test_cases"].append({
                "id": idx,
                "assignment": assignment,
                "prompt": prompt
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
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