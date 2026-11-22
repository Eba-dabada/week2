import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocess import load_data
from split import split_data

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
    
def get_prediction(model, loader, threshold=0.5):
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
        
def train_model(model, train_loader,val_loader, loss_fn, optimizer, epochs):
    train_losses = []
    val_losses = []
    for epoch in range (epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss =loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn)

        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
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

    y_train = train_loader.dataset.tensors[1]
    num_yes = y_train.sum().item()
    num_no = len(y_train) - num_yes
    pos_weight = torch.tensor([num_no / num_yes], dtype=torch.float32)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=100)

    preds, probs, y_true = get_prediction(model, val_loader)

    tp = ((preds == 1) & (y_true == 1)).sum().item()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()

    print(f"TP: {tp} | FP: {fp} | FN: {fn}")

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    plot_losses(train_losses, val_losses)