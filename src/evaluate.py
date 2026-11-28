import argparse
import json
import pathlib
import sys
import torch
import pandas as pd
from src.train import Bank_Marketing, get_metrics, get_prediction
from src.preprocess import fetch_data, load_data
from torch.utils.data import DataLoader
from src.split import BankDataset, get_split_indices


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print("Checkpoint keys:", list(checkpoint.keys()))
    model = Bank_Marketing(
        num_feature=checkpoint["num_features"],
        hidden_size=checkpoint["hidden_size"],
        dropout_rate=checkpoint["dropout_rate"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    df = load_data(fetch_data())
    indices = get_split_indices(df)
    test_df = df.loc[indices["test"]].copy()
    test_df["y"] = test_df["y"].map({"yes": 1, "no": 0})
    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]
    print("Test features shape:", X_test.shape)
    print("Positive count:", int(y_test.sum()))
    raw_dummies = pd.get_dummies(X_test)
    missing_cols = set(checkpoint["feature_names"])- set(raw_dummies.columns)
    print("Missing feature_names in test dummies:", missing_cols)
    X_test_encoded = raw_dummies.reindex(columns=checkpoint["feature_names"], fill_value=0)
    print("Encoded shape:", X_test_encoded.shape, "| Expected:", (len(X_test), checkpoint["num_features"]))
    X_test_float = X_test_encoded.astype(float)

    mean = torch.tensor(checkpoint["scaler_mean"], dtype=torch.float32)
    scale = torch.tensor(checkpoint["scaler_scale"], dtype=torch.float32)

    X_test_tensor = (torch.tensor(X_test_float.to_numpy(), dtype=torch.float32) - mean) / scale
    print("Scaled tensor shape:", X_test_tensor.shape, "| dtype:", X_test_tensor.dtype)
    
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    test_dataset = BankDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    threshold = checkpoint["threshold"]
    preds, probs, y_true = get_prediction(
        model,
        test_loader,
        threshold=threshold,
        device="cpu",
    )

    raw_metrics = get_metrics(y_true.numpy(), preds.numpy())

    output = {
        "test_f1": float(raw_metrics["f1"]),
        "test_accuracy": float(raw_metrics["accuracy"]),
        "test_precision": float(raw_metrics["precision"]),
        "test_recall": float(raw_metrics["recall"]),
        "threshold": float(threshold),
        "tp": int(raw_metrics["tp"]),
        "tn": int(raw_metrics["tn"]),
        "fp": int(raw_metrics["fp"]),
        "fn": int(raw_metrics["fn"]),
        "pred_yes": int(raw_metrics["pred_yes"]),
    }

    out_path = pathlib.Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf8")

    print("Saved test metrics:", output)
    return 0
if __name__ == "__main__":
    sys.exit(main())