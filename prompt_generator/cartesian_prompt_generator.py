#!/usr/bin/env python3
import copy
import jsonc
import itertools
import sys
import os

from typing import List, Dict, Tuple

class CartesianPromptGenerator:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = jsonc.load(f)

        split_attributes = self.config.get('split_attributes', [])
        split_attribute_labels = self.config.get('split_attribute_labels', [])
        merged_attributes = self.config.get('merged_attributes', [])
        merged_attribute_labels = self.config.get('merged_attribute_labels', [])

        self.factors = self.config['factors']
        self.template = self.config['base_prompt_template']
        self.factor_labels = [f['labels'] for f in self.factors]

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

        print(f"Cartesian Prompt Generator")
        print(f"Factors: {len(self.factors)}")
        for i, factor in enumerate(self.factors):
            print(f"  {i + 1}. {factor['name']}: {len(factor['labels'])} labels")
        print(f"\nTotal combinations (Full Cartesian): {self._get_full_size():,}")
        print(f"{'=' * 80}\n")

    def _get_full_size(self) -> int:
        size = 1
        for labels in self.factor_labels:
            size *= len(labels)
        return size

    def _generate_test_cases(self) -> List[Dict[str, str]]:
        """
        Generate test cases
        """
        test_cases = []
        combinations = list(itertools.product(*[f['labels'] for f in self.factors]))

        for combo in combinations:
            new_test_case = {}
            for i, cartesian_factor in enumerate(self.factors):
                new_test_case[cartesian_factor['name']] = combo[i]
            test_cases.append(new_test_case)

        print(f" total {len(test_cases)} test cases generated\n")
        return test_cases

    def generate_prompts(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Generate prompts incrementally
        """

        # Generate test cases
        test_cases = self._generate_test_cases()

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

                    merged_assignment[merged_attribute_name] = self.merged_attributes[merged_attribute_name][
                        split_attribute_index]
            try:
                prompt = self.template.format(**merged_assignment)
                prompts.append(prompt)
            except KeyError as e:
                raise ValueError(f"Template error: {e}")

        return prompts, test_cases

    def save_prompt_set(self, output_path: str):
        prompts, assignments = self.generate_prompts()

        output = {
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
            jsonc.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Total prompts: {len(prompts)}")
        return output

def main():
    if len(sys.argv) < 2:
        print("Usage: python cartesian_prompt_generator.py <config_json_path> [output_json_path]")
        print("\nExample:")
        print("python cartesian_prompt_generator.py prompt_config/cartesian_config.jsonc")
        sys.exit(1)

    config_path = sys.argv[1]

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        config_dir = os.path.dirname(config_path) if os.path.dirname(config_path) else '.'
        config_basename = os.path.basename(config_path)

        # Extract t value and create output filename
        if 'config' in config_basename:
            output_basename = config_basename.replace('config', 'prompts').replace('.jsonc', '.json').replace('.json',
                                                                                                              '.json')
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
    print(f"{'=' * 80}\n")

    generator = CartesianPromptGenerator(config_path)
    generator.save_prompt_set(output_path)

if __name__ == "__main__":
    main()