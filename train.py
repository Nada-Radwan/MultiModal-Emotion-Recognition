# train.py

import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.goemotions_loader import load_goemotions
from models.multimodal_emotion import TextEmotionModel
from utils.config import Config
from utils.metrics import compute_f1

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, best_f1, filename):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_f1": best_f1
    }, filename)


def train():

    device = Config.DEVICE
    dataset = load_goemotions()

    train_loader = DataLoader(
        dataset["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    model = TextEmotionModel().to(device)

    # ---------------------------
    # Compute class imbalance
    # ---------------------------
    print("Computing class weights...")
    all_labels = torch.stack([x["multi_labels"] for x in dataset["train"]])
    class_counts = all_labels.sum(dim=0)
    pos_weights = (len(all_labels) / (class_counts + 1e-6)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)

    writer = SummaryWriter("runs/goemotions")

    # ---------------------------
    # Resume Training If Exists
    # ---------------------------
    start_epoch = 0
    best_f1 = 0
    resume_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")

    if os.path.exists(resume_path):
        print("🔁 Resuming from checkpoint...")
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(start_epoch, Config.EPOCHS):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["multi_labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["multi_labels"].to(device)

                outputs = model(input_ids, attention_mask)

                all_preds.append(outputs)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        micro_f1, macro_f1 = compute_f1(all_preds, all_labels)

        print(f"Val F1 Micro: {micro_f1:.4f}")
        print(f"Val F1 Macro: {macro_f1:.4f}")

        writer.add_scalar("F1/micro", micro_f1, epoch)
        writer.add_scalar("F1/macro", macro_f1, epoch)

        # ---------------------------
        # Save Best Model
        # ---------------------------
        if micro_f1 > best_f1:
            best_f1 = micro_f1
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_f1,
                os.path.join(CHECKPOINT_DIR, "best_model.pt")
            )
            print("🔥 Saved BEST checkpoint!")

        # Save last checkpoint always
        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_f1,
            os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
        )

    writer.close()
    print("✅ Training finished.")


if __name__ == "__main__":
    train()
