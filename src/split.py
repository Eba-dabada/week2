import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from preprocess import load_data

def split_data(df):

    df["y"] = df["y"].map({"yes": 1, "no": 0})
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=0.60,
        random_state=42,
        stratify=y 
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=0.50,
        random_state=42,
        stratify=y_temp
    )

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
    X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor   = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)
    num_features = X_train_tensor.shape[1]
    feature_names = X_train_encoded.columns.tolist()
    return train_loader, val_loader, test_loader, num_features, feature_names

if __name__ == "__main__":
    path = r"C:\Users\Ivan\Desktop\week2\week2\bank-full.csv"

    df = load_data(path)
    train_loader, val_loader, test_loader, num_features, feature_names = split_data(df)
    
    batch_X, batch_y = next(iter(train_loader))
    print(f"Total input features: {num_features}")
    print(f"Batch X shape: {batch_X.shape} | dtype: {batch_X.dtype}")
    print(f"Batch y shape: {batch_y.shape} | dtype: {batch_y.dtype}")
    print(f"Total input features: {len(feature_names)}")
    print("\nAll encoded columns:")
    print(feature_names)


    unknown_cols = [col for col in feature_names if "unknown" in col]
    print(f"\nSurviving 'unknown' columns ({len(unknown_cols)}):")
    print(unknown_cols)

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size:   {len(val_loader.dataset)}")
    print(f"Test size:  {len(test_loader.dataset)}")

    train_y_mean = train_loader.dataset.tensors[1].mean().item()
    val_y_mean   = val_loader.dataset.tensors[1].mean().item()
    test_y_mean  = test_loader.dataset.tensors[1].mean().item()

    print(f"\nTrain positive class ratio: {train_y_mean:.4f}")
    print(f"Val positive class ratio:   {val_y_mean:.4f}")
    print(f"Test positive class ratio:  {test_y_mean:.4f}")