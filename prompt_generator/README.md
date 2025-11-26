# CT Prompt Generator README

## Usage

### Basic Usage

```bash
cd prompt_generator
python ct_prompt_generator.py <config_file_path> [output_file_path]
```

### Examples

```bash
cd prompt_generator

# Generate 2-wise prompts
python ct_prompt_generator.py prompt_config/ct_config_2wise.json prompt/ct_prompts_2wise.json

# Generate 3-wise prompts
python ct_prompt_generator.py prompt_config/ct_config_3wise.json prompt/ct_prompts_3wise.json

# Generate 4-wise prompts
python ct_prompt_generator.py prompt_config/ct_config_4wise.json prompt/ct_prompts_4wise.json
```

## Configuration File Format

Configuration files (`ct_config_*.json`) have the following structure:

```jsonc
{
  "t": 2,
  "base_prompt_template": "A photo of a {factor1} who is {factor2}",
  "cartesian_attributes": [...], // attribute that will do cartesian multiple
  "split_attributes": [
    [..., ...] // split attribute names
  ],
  "split_attribute_labels": [
    [..., ...] // split attribute label 1
    [..., ...] // split attribute label 2
    ...
  ],
  "merged_attributes": [
    ... // merged attribute name, index is same to split_attributes 
  ],
  "merged_attribute_labels": [
    ..., // merged attribute label 1
    ..., // merged attribute label 2
    ...
  ],
  "factors": [
    {
      "name": "factor1",
      "labels": [
        "label1",
        "label2",
        "label3"
      ]
    },
    {
      "name": "factor2",
      "labels": [
        "labelA",
        "labelB"
      ]
    }
  ]
}
```

- `t`: Coverage level (2, 3, 4, etc.)
- `base_prompt_template`: Prompt template (Python format string)
- `factors`: List of test factors and their corresponding labels

## Output File Format

```jsonc
{
  "t": 2,
  "base_prompt_template": "A photo of a {factor1} who is {factor2}",
  "factors": [...],
  "test_cases": [
    {
      "id": 0,
      "assignment": {
        "factor1": "label1",
        "factor2": "labelA"
      },
      "prompt": "A photo of a label1 who is labelA"
    },
    ...
  ]
}
```