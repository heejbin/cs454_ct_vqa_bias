## Requirements
*   Python 3.x
*   `scipy`
*   `numpy`

Install dependencies:
```bash
pip install scipy numpy
```

## Usage
```bash
python evaluator/run_chi_test.py <prompt_file_path> [--alpha <significance_level>]
```

## Input JSON Structure
The script expects the standard output format from the `ct_prompt_generator.py`:

```json
{
  "factors": [
    { "name": "gender", "labels": ["Male", "Female"], ... }
  ],
  "test_cases": [
    {
      "assignment": {
        "gender": "Male",
        "age": "Adult",
        ...
      },
      "prompt": "..."
    }
  ]
}
```

### Example Output
```text
Factor                         | Labels | Chi2 Stat  | P-value      | Result    
--------------------------------------------------------------------------------
gender                         | 2      | 4.1884     | 0.040701     | REJECT
socioeconomic_status           | 3      | 0.7826     | 0.676174     | ACCEPT
```