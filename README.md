## Finetune
* use alpaca prompt tuning
* modify huggingface dataset in finetune.py
```bash
python finetune.py
```

## Evaluation: HumanEval

### Installation

```bash
git clone https://github.com/openai/human-eval
cd human-eval/
pip install -e human-eval
```

### Generation

Generate samples and save them in the following JSON Lines (jsonl) format, where each sample is
formatted into a single line like so:

```
{"task_id": "Corresponding HumanEval task ID", "completion": "Completion only without the prompt"}
```

Run this code to generate the JSON files:
```bash
python eval_humaneval.py
```

### Evaluation

```bash
evaluate_functional_correctness samples.jsonl
```