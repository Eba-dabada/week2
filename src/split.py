import argparse
import json
import pathlib
import sys
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from src.preprocess import fetch_data, load_data


class BankDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_split_indices(df):
    y = df["y"]
    train_idx, temp_idx = train_test_split(
        df.index,
        train_size=0.60,
        random_state=42,
        stratify=y,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.50,
        random_state=42,
        stratify=y.loc[temp_idx],
    )
    return {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def split_data(df):
    df = df.copy()
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    X = df.drop(columns=["y"])
    y = df["y"]

    indices = get_split_indices(df)
    X_train, y_train = X.loc[indices["train"]], y.loc[indices["train"]]
    X_val, y_val = X.loc[indices["validation"]], y.loc[indices["validation"]]
    X_test, y_test = X.loc[indices["test"]], y.loc[indices["test"]]

    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, drop_first=True)
    X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train_encoded)
    X_test = scalar.transform(X_test_encoded)
    X_val = scalar.transform(X_val_encoded)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = BankDataset(X_train_tensor, y_train_tensor)
    val_dataset = BankDataset(X_val_tensor, y_val_tensor)
    test_dataset = BankDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    num_features = X_train_tensor.shape[1]
    feature_names = X_train_encoded.columns.tolist()
    return train_loader, val_loader, test_loader, num_features, feature_names, scalar


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    df = load_data(fetch_data())
    indices = get_split_indices(df)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(indices), encoding="utf8")

    return 0


if __name__ == "__main__":
    sys.exit(main())