"""Patient-level 5-fold teacher-student DenseNet distillation for TB/Cancer.

Required CSV: data/metadata.csv (image_path,label,patient_id)
Required teacher checkpoints: checkpoints/teacher/best_model_TB_cancer_fold{fold}.pth
"""
import copy
import json
import random
import time
from pathlib import Path

import nni
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from Distilation import densenet, Reduced_Desnsenet

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "metadata.csv"
DATA_ROOT = ROOT / "data"
TEACHER_DIR = ROOT / "checkpoints" / "teacher"
OUTPUT_ROOT = ROOT / "outputs"

CLASS_NAMES = ["TB", "Cancer"]  # TB=0, Cancer=1; NNI weight is applied to TB.
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE, NUM_FOLDS, NUM_EPOCHS = 224, 5, 20
NUM_WORKERS, SEED = 0, 0
USE_IMAGENET_PRETRAINING_FOR_STUDENT_INIT = True
TEMPERATURE_SQUARED_KD = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_CHECKPOINT_PATTERN = "best_model_TB_cancer_fold{fold}.pth"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(str(row.image_path))
        if not path.is_absolute():
            path = DATA_ROOT / path
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(
            CLASS_TO_INDEX[str(row.label)], dtype=torch.long
        )


def load_metadata():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CSV_PATH}; create metadata.csv with image_path,label,patient_id."
        )
    df = pd.read_csv(CSV_PATH)
    required = {"image_path", "label", "patient_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    if df[list(required)].eq("").any().any():
        raise ValueError("Empty metadata value found.")
    if set(df.label) - set(CLASS_NAMES):
        raise ValueError(f"Labels must be exactly {CLASS_NAMES}.")
    if df.image_path.duplicated().any():
        raise ValueError("Duplicate image_path entries.")
    if (df.groupby("patient_id").label.nunique() > 1).any():
        raise ValueError("Each patient must have exactly one class label.")
    for value in df.image_path:
        path = Path(value)
        if not path.is_absolute():
            path = DATA_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
    counts = df[["patient_id", "label"]].drop_duplicates().label.value_counts()
    for cls in CLASS_NAMES:
        if counts.get(cls, 0) < NUM_FOLDS:
            raise ValueError(f"Need at least {NUM_FOLDS} distinct {cls} patients.")
    print("Images per class:\n", df.label.value_counts().reindex(CLASS_NAMES, fill_value=0))
    print("Patients per class:\n", df[["patient_id", "label"]].drop_duplicates()
          .label.value_counts().reindex(CLASS_NAMES, fill_value=0))
    return df


def make_transforms():
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf, tf


def split_train_validation_by_patient(outer_train, valid_size):
    patients = outer_train[["patient_id", "label"]].drop_duplicates()
    if valid_size <= 0 or valid_size >= len(patients):
        raise ValueError(
            f"valid_size={valid_size} must be between 1 and {len(patients)-1} "
            "validation patients in this outer-training fold."
        )
    train_ids, valid_ids = train_test_split(
        patients.patient_id.to_numpy(),
        test_size=valid_size,
        random_state=SEED,
        stratify=patients.label.to_numpy(),
    )
    train_df = outer_train[outer_train.patient_id.isin(train_ids)].copy()
    valid_df = outer_train[outer_train.patient_id.isin(valid_ids)].copy()
    assert set(train_df.patient_id).isdisjoint(set(valid_df.patient_id))
    if set(train_df.label) != set(CLASS_NAMES) or set(valid_df.label) != set(CLASS_NAMES):
        raise ValueError("Both classes must occur in train and validation patients.")
    return train_df, valid_df


def load_teacher(fold):
    path = TEACHER_DIR / TEACHER_CHECKPOINT_PATTERN.format(fold=fold)
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    # weights_only=False supports old torch.save(model, path) files.
    # Only load trusted checkpoints.
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, nn.Module):
        teacher = checkpoint
    else:
        teacher = densenet.densenet121(pretrained=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        teacher.load_state_dict(state, strict=True)
    teacher = teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def build_student_initial_state(dense3, dense4):
    pretrained = densenet.densenet121(
        pretrained=USE_IMAGENET_PRETRAINING_FOR_STUDENT_INIT
    )
    student = Reduced_Desnsenet.densenet121(
        dense_3=int(dense3), dense_4=int(dense4)
    )
    source, target = pretrained.state_dict(), student.state_dict()
    excluded = ("transition3", "classifier", "norm5", "denseblock4")
    copied = []
    for key, value in source.items():
        if any(token in key for token in excluded):
            continue
        if key in target and target[key].shape == value.shape:
            target[key].copy_(value)
            copied.append(key)
    if not copied:
        raise RuntimeError(
            "No compatible pretrained weights copied. Check the two DenseNet modules."
        )
    student.load_state_dict(target)
    del pretrained
    if not hasattr(student, "classifier"):
        raise AttributeError("Reduced DenseNet must expose model.classifier.")
    student.classifier = nn.Linear(student.classifier.in_features, 2)
    return student.to(DEVICE), copied


def metrics_for(labels, predictions):
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    report = classification_report(
        labels, predictions, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    recalls = recall_score(labels, predictions, labels=[0, 1],
                           average=None, zero_division=0)
    return cm, report, float(recalls[0] * recalls[1])


def train_epoch(student, teacher, loader, kd_criterion, ce_criterion,
                optimizer, temperature, kd_weight, ce_weight):
    student.train()
    teacher.eval()
    total_loss = total_correct = total_count = 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)

        teacher_prob = torch.softmax(teacher_logits / temperature, dim=1)
        student_log_prob = torch.log_softmax(student_logits / temperature, dim=1)
        kd_loss = kd_criterion(student_log_prob, teacher_prob)
        if TEMPERATURE_SQUARED_KD:
            kd_loss = kd_loss * temperature ** 2
        ce_loss = ce_criterion(student_logits, labels)
        loss = kd_weight * kd_loss + ce_weight * ce_loss
        loss.backward()
        optimizer.step()

        pred = student_logits.argmax(dim=1)
        n = labels.size(0)
        total_loss += loss.item() * n
        total_correct += (pred == labels).sum().item()
        total_count += n
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    return total_loss / max(total_count, 1), 100 * total_correct / max(total_count, 1), y_true, y_pred


@torch.no_grad()
def evaluate(student, loader, ce_criterion):
    student.eval()
    loss_sum = 0.0
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = student(images)
        loss_sum += ce_criterion(logits, labels).item() * labels.size(0)
        pred = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    cm, report, recall_product = metrics_for(y_true, y_pred)
    return {
        "loss": loss_sum / max(len(loader.dataset), 1),
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "labels": y_true,
        "predictions": y_pred,
        "confusion_matrix": cm,
        "report": report,
        "recall_product": recall_product,
    }


def run_fold(fold, outer_train, test_df, params, trial_id):
    batch = int(params["train_batch"])
    valid_size = int(params["valid_size"])
    lr = float(params["lr"])
    tb_weight = float(params["weight"])
    dense3, dense4 = int(params["dense3"]), int(params["dense4"])
    temperature = float(params["temp"])
    kd_weight = float(params["weight_loss1"])
    ce_weight = float(params["weight_loss2"])

    train_df, valid_df = split_train_validation_by_patient(outer_train, valid_size)
    train_tf, eval_tf = make_transforms()
    train_loader = DataLoader(
        BronchoscopyDataset(train_df, train_tf), batch_size=batch,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    valid_loader = DataLoader(
        BronchoscopyDataset(valid_df, eval_tf), batch_size=min(batch, 32),
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        BronchoscopyDataset(test_df, eval_tf), batch_size=min(batch, 32),
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    teacher = load_teacher(fold)
    student, copied_keys = build_student_initial_state(dense3, dense4)
    initial_state = copy.deepcopy(student.state_dict())

    # TB is index 0: NNI's weight parameter applies to TB, Cancer stays at 1.
    class_weights = torch.tensor([tb_weight, 1.0], dtype=torch.float32, device=DEVICE)
    class_weights = class_weights / class_weights.sum()
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    kd_criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(student.parameters(), lr=lr)

    print(
        f"Fold {fold}: train/valid/test patients = "
        f"{train_df.patient_id.nunique()}/{valid_df.patient_id.nunique()}/"
        f"{test_df.patient_id.nunique()}; params={params}"
    )
    print(f"Transferred {len(copied_keys)} compatible pretrained tensors.")

    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(initial_state)

    for epoch in range(1, 21):
        tr_loss, tr_acc, _, _ = train_epoch(
            student, teacher, train_loader, kd_criterion, ce_criterion,
            optimizer, temperature, kd_weight, ce_weight
        )
        val = evaluate(student, valid_loader, ce_criterion)
        print(
            f"Fold {fold} Epoch {epoch}/20: train loss={tr_loss:.4f}, "
            f"acc={tr_acc:.2f}% | val loss={val['loss']:.4f}, "
            f"acc={val['accuracy']:.2f}%, TB×Cancer recall={val['recall_product']:.5f}"
        )
        # Preserve the original best-epoch selection criterion: validation accuracy.
        if val["accuracy"] > best_val_accuracy:
            best_val_accuracy = val["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    # Evaluate the selected best-validation-accuracy checkpoint, not the final epoch.
    best_validation = evaluate(student, valid_loader, ce_criterion)
    test = evaluate(student, test_loader, ce_criterion)

    out = OUTPUT_ROOT / f"trial_{trial_id}" / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "class_names": CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "hyperparameters": params,
        "fold": fold,
    }, out / "best_student_model.pth")

    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_validation_accuracy": float(best_val_accuracy),
        "validation_recall_product": float(best_validation["recall_product"]),
        "test_loss": float(test["loss"]),
        "test_accuracy": float(test["accuracy"]),
        "test_recall_product": float(test["recall_product"]),
        "tb_recall": float(test["report"]["TB"]["recall"]),
        "cancer_recall": float(test["report"]["Cancer"]["recall"]),
        "test_patients": int(test_df.patient_id.nunique()),
        "confusion_matrix": test["confusion_matrix"].tolist(),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Test confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(test["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES))

    del teacher, student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, float(best_validation["recall_product"])


def main():
    set_seed()
    received = nni.get_next_parameter()
    params = {
        "dense3": int(received["dense3"]),
        "dense4": int(received["dense4"]),
        "train_batch": int(received["train_batch"]),
        "valid_size": int(received["valid_size"]),
        "lr": float(received["lr"]),
        "weight": int(received["weight"]),
        "temp": int(received["temp"]),
        "weight_loss1": int(received["weight_loss1"]),
        "weight_loss2": int(received["weight_loss2"]),
    }
    trial_id = nni.get_trial_id()
    print("Trial:", trial_id, "| Device:", DEVICE, "| Parameters:", params)

    df = load_metadata()
    y = df.label.map(CLASS_TO_INDEX).to_numpy()
    groups = df.patient_id.to_numpy()
    cv = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    results, validation_objectives = [], []
    started = time.time()
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(df)), y, groups), start=1
    ):
        outer_train = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        if not set(outer_train.patient_id).isdisjoint(set(test_df.patient_id)):
            raise RuntimeError("Patient leakage between outer train and test.")
        fold_result, val_objective = run_fold(
            fold, outer_train, test_df, params, trial_id
        )
        results.append(fold_result)
        validation_objectives.append(val_objective)
        nni.report_intermediate_result(float(np.mean(validation_objectives)))

    mean_val_objective = float(np.mean(validation_objectives))
    mean_test_accuracy = float(np.mean([r["test_accuracy"] for r in results]))
    mean_test_product = float(np.mean([r["test_recall_product"] for r in results]))

    out = OUTPUT_ROOT / f"trial_{trial_id}"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "trial_id": trial_id,
        "hyperparameters": params,
        "class_names": CLASS_NAMES,
        "nni_objective": "mean validation TB recall multiplied by Cancer recall",
        "mean_validation_recall_product": mean_val_objective,
        "mean_test_accuracy_reporting_only": mean_test_accuracy,
        "mean_test_recall_product_reporting_only": mean_test_product,
        "fold_results": results,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - started
    print(f"Mean validation TB×Cancer recall: {mean_val_objective:.6f}")
    print(f"Mean test accuracy (reporting only): {mean_test_accuracy:.2f}%")
    print(f"Mean test recall product (reporting only): {mean_test_product:.6f}")
    print(f"Training and testing complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    nni.report_final_result(mean_val_objective)


if __name__ == "__main__":
    main()
