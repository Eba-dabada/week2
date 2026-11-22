from sklearn.ensemble import HistGradientBoostingClassifier
from preprocess import load_data
from split import split_data


if __name__ == "__main__":
    path = r"C:\Users\Ivan\Desktop\week2\week2\bank-full.csv"
    df = load_data(path)
    train_loader, val_loader, _, _, _ = split_data(df)

    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy().ravel()

    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy().ravel()

    print("X train:", X_train.shape)
    print("y train:", y_train.shape)
    print("X val:  ", X_val.shape)
    print("y val:  ", y_val.shape)

    clf = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
    print(clf)

    clf.fit(X_train, y_train)
    print("Classes:", clf.classes_)
    print("Iterations:", clf.n_iter_)

    probs_all = clf.predict_proba(X_val)

    print("probs_all shape:", probs_all.shape)
    print("First 3 rows:\n", probs_all[:3])

    probs = probs_all[:, 1]

    preds = (probs >= 0.5).astype(int)

    print("probs shape:", probs.shape)
    print("preds shape:", preds.shape)
    print("Predicted 'yes' count:", preds.sum())

    tp = ((preds == 1) & (y_val == 1)).sum().item()
    fp = ((preds == 1) & (y_val == 0)).sum().item()
    fn = ((preds == 0) & (y_val == 1)).sum().item()

    print(f"TP: {tp} | FP: {fp} | FN: {fn}")

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")    