"""Optional test against a real dataset, computing metrics from held-out data."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest

def test_real_chronological_holdout():
    path = os.environ.get('FRAUD_DATASET_PATH')
    if not path:
        pytest.skip('Set FRAUD_DATASET_PATH to the credit-card CSV')
    spec = importlib.util.spec_from_file_location('train', Path(__file__).parents[1] / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    X, y = train.load_kaggle_dataset(Path(path), max_samples=20_000)
    split = int(len(y) * .8)
    mean, scale = X[:split].mean(axis=0), X[:split].std(axis=0)
    scale[scale == 0] = 1
    w, b = train.train_logistic_regression((X[:split] - mean) / scale, y[:split], epochs=250)
    metrics = train.evaluate_model((X[split:] - mean) / scale, y[split:], w, b)
    assert metrics['test_samples'] == len(y) - split
    assert metrics['fraud_cases'] == int(y[split:].sum()) > 0
    assert all(np.isfinite(metrics[k]) and 0 <= metrics[k] <= 1 for k in ('precision', 'recall', 'f1'))
    print('Measured holdout:', metrics)
