import argparse
import pathlib
import sys
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from src.preprocess import load_data, fetch_data
from src.split import split_data
from torch.utils.data import DataLoader, Subset

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

def evaluate(model, loader, loss_fn, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

def get_prediction(model, loader, threshold=0.7, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_probs =[]
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= threshold).float()

            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(batch_y)

    all_probs = torch.cat(all_probs).squeeze()
    all_preds = torch.cat(all_preds).squeeze()
    all_targets = torch.cat(all_targets).squeeze()

    return all_preds, all_probs, all_targets

def select_best_threshold(model, val_loader, device, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    best_threshold = 0.7
    best_metrics = None
    best_f1 = -1.0

    for threshold in thresholds:
        preds, _, y_true = get_prediction(model, val_loader,threshold=threshold, device=device)
        metrics = get_metrics(y_true.numpy(), preds.numpy())
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics

def save_checkpoint(path, model, optimizer, epoch, best_metric, history, metadata=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "history": history,
    }
    if metadata:
        checkpoint.update(metadata)
    torch.save(checkpoint, path)
    

def load_checkpoint(path, model, optimizer, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (checkpoint.get("epoch", 0), checkpoint.get("best_metric"), checkpoint.get("history",{}))
        
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, checkpoint_path="best_model.pt", device=None, metadata=None):
    if device is None:
        device = next(model.parameters()).device

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
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
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=best_epoch,
                best_metric=best_val_loss,
                history={"train_losses": train_losses, "val_losses": val_losses},
                metadata=metadata,
            )

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"Best model loaded from epoch {best_epoch} with validation loss {best_val_loss:.4f}")
    return train_losses, val_losses, best_epoch, best_val_loss


def plot_losses(train_losses, val_losses, save_path=None):
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

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")

    plt.close()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data(fetch_data())
    train_loader, val_loader, test_loader, num_features, feature_names, scalar = split_data(df)
    epochs = args.epochs
    if args.smoke:
        epochs = min(epochs, 2)
        smoke_size = min(512, len(train_loader.dataset))
        smoke_subset = Subset(train_loader.dataset, range(smoke_size))
        train_loader = DataLoader(smoke_subset, batch_size= train_loader.batch_size, shuffle=True)
        y_train = train_loader.dataset.dataset.y[smoke_subset.indices].numpy().ravel()
    else:
        y_train = train_loader.dataset.y.numpy().ravel()
    
    hidden_size = 41
    dropout_rate = 0.3
    model = Bank_Marketing(num_feature=num_features, hidden_size=hidden_size, dropout_rate=dropout_rate).to(device)
    
    num_yes = max(y_train.sum().item(), 1.0)
    num_no = len(y_train) - y_train.sum().item()
    pos_weight = torch.tensor([num_no / num_yes], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_features": num_features,
        "hidden_size": hidden_size,
        "dropout_rate": dropout_rate,
        "feature_names": feature_names,
        "scaler_mean": scalar.mean_.tolist(),
        "scaler_scale": scalar.scale_.tolist(),
        "threshold": 0.7,
    }
    
    train_losses, val_losses, best_epoch, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        epochs=epochs,
        checkpoint_path=out_path,
        device=device,
        metadata=metadata,
    )

    best_threshold, val_metrics = select_best_threshold(model, val_loader, device=device)
    metadata["threshold"] = best_threshold

    save_checkpoint(
        out_path,
        model,
        optimizer,
        epoch=best_epoch,
        best_metric=best_val_loss,
        history={"train_losses": train_losses, "val_losses": val_losses},
        metadata=metadata,
    )

    print(f"Chosen threshold on validation: {best_threshold:.2f} | Val F1: {val_metrics['f1']:.4f}")
    
    if args.plot and not args.smoke:
        plot_losses(train_losses, val_losses, save_path=out_path.with_suffix(".png"))
    return 0

if __name__ == "__main__":
    sys.exit(main())