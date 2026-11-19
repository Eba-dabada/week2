import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

if __name__ == "__main__":
    path = r"C:\Users\EayeR\Documents\week 2\bank-full.csv"

    df = load_data(path)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    print("\n--- PyTorch Tensor Shapes Inspection ---")
    print(f"X_train shape: {X_train.shape} | Expected: (Rows, Features)")
    print(f"y_train shape: {y_train.shape} | Expected: (Rows, 1)")
    print("-" * 40)
    print(f"X_val shape:   {X_val.shape}")
    print(f"y_val shape:   {y_val.shape}")
    print("-" * 40)
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")

