from json import dumps
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (BertForMultipleChoice, BertTokenizer, Trainer,
                          set_seed)

from util import get_dataset, get_trainer


def bert_predict(logits: np.ndarray) -> List[Tuple[int, float]]:
    log_softmax: torch.Tensor = torch.tensor(logits).log_softmax(1)
    predictions: List[Tuple[int, float]] = []

    for x in log_softmax:
        prediction: int = torch.argmax(x).item()
        predictions.append((prediction + 1, x[prediction].item()))

    return predictions


if __name__ == '__main__':
    set_seed(42)
    model_path = 'model'
    Path('output').mkdir(exist_ok=True)

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)

    model: BertForMultipleChoice = BertForMultipleChoice.from_pretrained(
        model_path
    )

    trainer: Trainer = get_trainer(None, None, tokenizer, model)

    in_file: str = 'data/test_entry.jsonl'
    out_file: str = 'output/predict.jsonl'
    test_set: Dataset = get_dataset(in_file, tokenizer)

    bert_pred: List[Tuple[int, float]] = bert_predict(
        trainer.predict(test_set).predictions
    )

    with open(out_file, 'w', encoding='utf8') as f:
        for i, id in enumerate(test_set['id']):
            print(dumps({
                'id': id,
                'answer': bert_pred[i][0]
            }), file=f)
