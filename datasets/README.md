# Datasets

## Code execution tasks

The script `finetuning/sft/prepare_code_execution_data.py` collects prompt/solution pairs from open-source repositories. It scans Python files for comments formatted as:

```
# prompt: describe the task
# solution: expected answer
```

### Usage

Run the script with a list of repository URLs and an output path:

```
python finetuning/sft/prepare_code_execution_data.py --repos https://github.com/user/repo1 https://github.com/user/repo2 --output datasets/code_tasks.jsonl
```

Repositories are cloned into the `external_repos` directory by default. The resulting JSONL file contains entries with `prompt` and `solution` keys suitable for fine-tuning.
