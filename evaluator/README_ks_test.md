
## Requirements
*   Python 3.x
*   `scipy`
*   `numpy`
*   `matplotlib`

Install dependencies:
```bash
pip install scipy numpy matplotlib
```

## Usage
```bash
python evaluator/run_ks_test.py <file1_path> <file2_path> [--label1 <label>] [--label2 <label>] [--output_dir <directory>]
```

### Arguments and Options
*   `<file1_path>`: Path to the first VQA scores JSON file.
*   `<file2_path>`: Path to the second VQA scores JSON file.
*   `--label1 <label>`: Optional label for the first file (default: filename).
*   `--label2 <label>`: Optional label for the second file (default: filename).
*   `--output_dir <directory>`: Optional root directory for results (default: `ks_results`).

## Input JSON Structure

```json
[
    {
        "assigned_attributes": { "race_ethnicity": "South-Asian", ... },
        "vqa_results": {
            "race_ethnicity": { "yes_probability": 0.999, ... },
            // ... other attribute results
        },
        "summary": { "average_yes_probability": 0.662, ... }
    }
]
```

## Output Structure
Results are saved in the specified `--output_dir`.

*   `ks_results/ks_summary.csv`:
    *   CSV table of all KS test results.
    *   Columns: `Attribute`, `Value`, `KS_Statistic`, `P_Value`, `[Label1]_Count`, `[Label2]_Count`, `Type`.
    *   `Type` is either `value` (individual attribute) or `theme` (aggregated attribute).

*   `ks_results/plots/`:
    *   Subdirectories per `Attribute` (e.g., `plots/appearance_body_status/`).
    *   Each subdirectory contains ECDF comparison plots (`.png`) for each `Value` (e.g., `Thin.png`).