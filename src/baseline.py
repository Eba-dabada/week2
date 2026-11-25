from sklearn.ensemble import HistGradientBoostingClassifier
from src.preprocess import load_data
from src.split import split_data
import numpy as np

def get_matrics(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return accuracy, precision, recall, f1

def trivial(y_train, shape):
    y_arr = np.asarray(y_train)
    fill_value = int(y_arr.mean() > 0.5)
    return np.full(shape, fill_value)

if __name__ == "__main__":
    path = r"C:\Users\Ivan\Desktop\week2\week2\bank-full.csv"
    df = load_data(path)
    train_loader, val_loader, _, _, _ = split_data(df)

    X_train = train_loader.dataset.X.numpy()
    y_train = train_loader.dataset.y.numpy().ravel()

    X_val = val_loader.dataset.X.numpy()
    y_val = val_loader.dataset.y.numpy().ravel()

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


    accuracy, precision, recall, f1 = get_matrics(y_val, preds)
    matrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    print("Metrics:", matrics)

    trivial_preds = trivial(y_train, y_val.shape)

    print("Trivial preds shape:", trivial_preds.shape, "| y_val.shape:", y_val.shape)
    print("Trivial predicted 'yes' count:", trivial_preds.sum())

    t_acc, t_prec, t_rec, t_f1 = get_matrics(y_val, trivial_preds)
    trivial_metrics = {
        "accuracy": t_acc,
        "precision": t_prec,
        "recall": t_rec,
        "f1": t_f1,
    }
    print("Trivial (majority) metrics:", trivial_metrics)