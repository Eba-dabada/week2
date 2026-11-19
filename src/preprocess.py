import pandas as pd
import torch

def load_data(path):
    df = pd.read_csv(path, sep=";")
    df.drop(columns=["duration"], inplace=True)
    return df

if __name__ == "__main__":
    path = r"C:\Users\EayeR\Documents\week 2\bank-full.csv"
    df = load_data(path)
    
    print(df.head())
    print("\n--- Data Loaded Directly ---")
    print(df.shape)
    print(df["y"].value_counts(normalize=True))