import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from src.preprocess import load_data
from src.split import split_data

class Bank_Marketing(nn.Module):
    def __init__(self, num_feature, hidden_size=41, dropout_rate=0.3):
        super(Bank_Marketing, self).__init__()
        self.z1 = nn.Linear(num_feature, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.z2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.relu(self.z1(x))
        x = self.dropout(x)
        x = self.z2(x)
        return x

def get_metrics(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_yes": tp + fp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

def get_prediction(model, loader, threshold=0.6):
    model.eval()
    all_preds = []
    all_probs =[]
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(batch_y)

    all_probs = torch.cat(all_probs).squeeze()
    all_preds = torch.cat(all_preds).squeeze()
    all_targets = torch.cat(all_targets).squeeze()

    return all_preds, all_probs, all_targets

def save_checkpoint(path, model, optimizer, epoch, best_metric, history):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "history": history,
    }
    torch.save(checkpoint, path)
    

def load_checkpoint(path, model, optimizer, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (checkpoint.get("epoch", 0), checkpoint.get("best_metric"), checkpoint.get("history",{}))
        
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, checkpoint_path="best_model.pt"):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn)

        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, checkpoint_path)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"Best model loaded from epoch {best_epoch} with validation loss {best_val_loss:.4f}")
    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    epoch_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_range, train_losses, label="Train Loss", marker="o", markersize=4)
    plt.plot(epoch_range, val_losses, label="Validation Loss", marker="s", markersize=4)

    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    path = r"C:\Users\Ivan\Desktop\week2\week2\bank-full.csv"
    df = load_data(path)
    train_loader, val_loader, test_loader, num_features, feature_names = split_data(df)

    model = Bank_Marketing(num_feature=num_features)

    y_train = train_loader.dataset.y.numpy().ravel()
    num_yes = y_train.sum().item()
    num_no = len(y_train) - num_yes
    pos_weight = torch.tensor([num_no / num_yes], dtype=torch.float32)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=2)

    preds, probs, y_true = get_prediction(model, val_loader, threshold=0.6)
    metrics = get_metrics(y_true.numpy(), preds.numpy())
    print("Final threshold 0.6 Metrics:", metrics)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds, probs, y_true = get_prediction(model, val_loader, threshold=threshold)
        metrics = get_metrics(y_true.numpy(), preds.numpy())
        print(f"Threshold {threshold:.2f} Metrics:", metrics)

    plot_losses(train_losses, val_losses)