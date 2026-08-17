"""
data.py
=======
Creates the three synthetic classification datasets (Small / Medium / Large)
used to study how dataset size interacts with batch size.
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_SIZES = {
    "Small (1K)": 1_000,
    "Medium (10K)": 10_000,
    "Large (50K)": 50_000,
}


def _make_dataset(n_samples, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def prepare_datasets(verbose=True):
    """
    Build the Small / Medium / Large synthetic datasets.

    Returns
    -------
    datasets : dict[str, tuple]
        name -> (X_train, y_train, X_test, y_test)
    datasets_info : pandas.DataFrame
        Summary table (train/test sizes, feature count, class count).
    """
    if verbose:
        print("\n PREPARING DATASETS OF DIFFERENT SIZES...")

    datasets = {}
    for name, n_samples in DATASET_SIZES.items():
        if verbose:
            print(f"\n Creating {name.split(' ')[0].upper()} dataset ({n_samples:,} samples)...")
        datasets[name] = _make_dataset(n_samples)

    datasets_info = pd.DataFrame(
        {
            "Dataset": [n.split(" ")[0] for n in datasets.keys()],
            "Train Samples": [len(v[0]) for v in datasets.values()],
            "Test Samples": [len(v[2]) for v in datasets.values()],
            "Features": [20] * len(datasets),
            "Classes": [2] * len(datasets),
        }
    )

    if verbose:
        print("\n Dataset Summary:")
        print(datasets_info.to_string(index=False))

    return datasets, datasets_info
