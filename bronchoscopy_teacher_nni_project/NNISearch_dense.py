"""NNI trial: patient-level 5-fold DenseNet-121 classification.

CSV: data/metadata.csv with columns image_path,label,patient_id.
Labels: TB and Cancer. Each patient must have one class.
NNI optimizes mean best validation accuracy (not test accuracy).
valid_size means number of validation PATIENTS in each fold.
"""
import copy, json, random
from pathlib import Path

import nni
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

CSV_PATH = Path("data/metadata.csv")
DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("outputs")
CLASS_NAMES = ["TB", "Cancer"]  
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE, NUM_FOLDS, NUM_EPOCHS = 224, 5, 20
NUM_WORKERS, SEED = 0, 42
USE_PRETRAINED, WEIGHT_DECAY = True, 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BronchoscopyDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = Path(str(row.image_path))
        if not path.is_absolute():
            path = DATA_ROOT / path
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(
            CLASS_TO_INDEX[str(row.label)], dtype=torch.long
        )


def load_metadata():
    df = pd.read_csv(CSV_PATH)
    required = {"image_path", "label", "patient_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    for c in required:
        df[c] = df[c].astype(str).str.strip()
    if df[list(required)].eq("").any().any():
        raise ValueError("Empty metadata value found.")
    if set(df.label) - set(CLASS_NAMES):
        raise ValueError(f"Labels must be exactly {CLASS_NAMES}.")
    if df.image_path.duplicated().any():
        raise ValueError("Duplicate image_path entries.")
    if (df.groupby("patient_id").label.nunique() > 1).any():
        raise ValueError("A patient has multiple class labels; inspect metadata.")
    for value in df.image_path:
        p = Path(value)
        if not p.is_absolute():
            p = DATA_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(p)
    patient_counts = df[["patient_id", "label"]].drop_duplicates().label.value_counts()
    for cls in CLASS_NAMES:
        if patient_counts.get(cls, 0) < NUM_FOLDS:
            raise ValueError(f"Need at least {NUM_FOLDS} distinct {cls} patients.")
    return df


def transforms_for_data():
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    return (
        transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                            transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                            transforms.ToTensor(), norm])
    )


def build_model():
    weights = models.DenseNet121_Weights.DEFAULT if USE_PRETRAINED else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model.to(DEVICE)


def split_train_valid(outer_train, valid_size):
    # Split unique patients, then map patient IDs back to all their images.
    patients = outer_train[["patient_id", "label"]].drop_duplicates()
    if valid_size <= 0 or valid_size >= len(patients):
        raise ValueError(
            f"valid_size={valid_size} must be between 1 and "
            f"{len(patients)-1} validation patients."
        )
    train_ids, valid_ids = train_test_split(
        patients.patient_id.to_numpy(), test_size=valid_size,
        random_state=SEED, stratify=patients.label.to_numpy()
    )
    train_df = outer_train[outer_train.patient_id.isin(train_ids)].copy()
    valid_df = outer_train[outer_train.patient_id.isin(valid_ids)].copy()
    assert set(train_df.patient_id).isdisjoint(set(valid_df.patient_id))
    if set(train_df.label) != set(CLASS_NAMES) or set(valid_df.label) != set(CLASS_NAMES):
        raise ValueError("Both classes must occur in train and validation patients.")
    return train_df, valid_df


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    loss_sum, truth, predictions_all = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * len(images)
        truth.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(1).detach().cpu().tolist())
    return loss_sum / len(loader.dataset), accuracy_score(truth, predictions_all)*100, truth, predictions_all


def run_fold(fold, outer_train, test_df, params, trial_id):
    batch = int(params["train_batch"])
    valid_size = int(params["valid_size"])
    lr, weight = float(params["lr"]), int(params["weight"])
    train_df, valid_df = split_train_valid(outer_train, valid_size)
    train_tf, eval_tf = transforms_for_data()

    train_loader = DataLoader(BronchoscopyDataset(train_df, train_tf),
        batch_size=batch, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(BronchoscopyDataset(valid_df, eval_tf),
        batch_size=min(batch, 32), shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(BronchoscopyDataset(test_df, eval_tf),
        batch_size=min(batch, 32), shuffle=False, num_workers=NUM_WORKERS)

    model = build_model()
    # Class index 0 is TB, so the tunable weight is applied to TB.
    class_weights = torch.tensor([float(weight), 1.0], device=DEVICE)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_val, best_epoch = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    print(f"Fold {fold}: train/valid/test patients = "
          f"{train_df.patient_id.nunique()}/{valid_df.patient_id.nunique()}/"
          f"{test_df.patient_id.nunique()}; params={params}")

    for epoch in range(1, NUM_EPOCHS+1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, valid_loader, criterion)
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"Fold {fold} Epoch {epoch}/{NUM_EPOCHS}: "
              f"train loss={tr_loss:.4f}, acc={tr_acc:.2f}% | "
              f"valid loss={va_loss:.4f}, acc={va_acc:.2f}%")

    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1],
        target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    out = OUTPUT_ROOT / f"trial_{trial_id}" / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "class_names": CLASS_NAMES,
        "best_epoch": best_epoch, "best_validation_accuracy": best_val,
        "hyperparameters": params}, out / "best_model.pth")
    result = {
        "fold": fold, "best_epoch": best_epoch,
        "best_validation_accuracy": best_val, "test_loss": test_loss,
        "test_accuracy": test_acc, "test_patients": int(test_df.patient_id.nunique()),
        "cancer_recall": report["Cancer"]["recall"], "tb_recall": report["TB"]["recall"],
        "confusion_matrix": cm.tolist()
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Test confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))
    return result


def main():
    set_seed()
    received = nni.get_next_parameter()
    params = {
        "train_batch": int(received["train_batch"]),
        "valid_size": int(received["valid_size"]),
        "lr": float(received["lr"]),
        "weight": int(received["weight"])
    }
    trial_id = nni.get_trial_id()
    print("Trial:", trial_id, "| Device:", DEVICE, "| Parameters:", params)

    df = load_metadata()
    y = df.label.map(CLASS_TO_INDEX).to_numpy()
    groups = df.patient_id.to_numpy()
    cv = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_results, validation_scores = [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(df)), y, groups), 1):
        outer_train, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        if not set(outer_train.patient_id).isdisjoint(set(test_df.patient_id)):
            raise RuntimeError("Patient leakage between outer train and test.")
        result = run_fold(fold, outer_train, test_df, params, trial_id)
        fold_results.append(result)
        validation_scores.append(result["best_validation_accuracy"])

    mean_valid = float(np.mean(validation_scores))
    mean_test = float(np.mean([x["test_accuracy"] for x in fold_results]))
    out = OUTPUT_ROOT / f"trial_{trial_id}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"trial_id": trial_id, "hyperparameters": params,
            "mean_best_validation_accuracy": mean_valid,
            "mean_test_accuracy_reporting_only": mean_test,
            "fold_results": fold_results}, f, indent=2)
    print(f"Mean best validation accuracy: {mean_valid:.2f}%")
    print(f"Mean test accuracy (reporting only): {mean_test:.2f}%")
    # NNI tuner must optimize validation results, not held-out test results.
    nni.report_final_result(mean_valid)


if __name__ == "__main__":
    main()
