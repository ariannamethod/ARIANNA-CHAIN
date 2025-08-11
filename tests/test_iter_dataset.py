import json

from finetuning.grpo_train import iter_dataset


def test_iter_dataset_large_file(tmp_path):
    dataset = tmp_path / "large.jsonl"
    with dataset.open("w", encoding="utf-8") as f:
        for i in range(10000):
            conf = 0.9 if i % 2 == 0 else 0.1
            obj = {"prompt": f"Q{i}", "answer": f"A{i}", "confidence": conf}
            f.write(json.dumps(obj) + "\n")
    count = 0
    for sample in iter_dataset(dataset, min_confidence=0.5):
        if count == 0:
            assert sample["prompt"] == "Q0" and sample["answer"] == "A0"
        count += 1
    assert count == 5000
