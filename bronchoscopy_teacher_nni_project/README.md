# Patient-level Bronchoscopy DenseNet + NNI

## Dataset
Create `data/metadata.csv` with columns `image_path,label,patient_id`.
Labels must be exactly `TB` or `Cancer`. Use the same patient_id for every
image from the same patient. Each patient must have only one class label.

Example:
```csv
image_path,label,patient_id
images/TB/p001_01.jpg,TB,P001
images/TB/p001_02.jpg,TB,P001
images/Cancer/p002_01.jpg,Cancer,P002
```
Paths are relative to `data/` unless absolute.

## Run
```bash
pip install -r requirements.txt
python run_experiment.py
```
NNI dashboard: http://localhost:8080

## Files
- `Search_space.py`: search-space dictionary
- `run_experiment.py`: NNI experiment launcher
- `NNISearch_dense.py`: training trial with patient-level five-fold CV

## Search parameters
- train_batch: mini-batch size
- valid_size: number of validation patients in each fold, NOT image count
- lr: Adam learning rate
- weight: Cancer weight in normalized `[weight, 1]` class weights

The class order is Cancer=0, TB=1. This preserves torchvision ImageFolder's
alphabetical folder order and the original `[weight, 1]` weighting convention.

NNI optimizes mean best validation accuracy across folds. Test results are
saved for reporting but are not used as the tuner objective.

## Resource warning
Concurrency 10 can launch ten DenseNet jobs simultaneously and may exhaust
GPU memory or RAM. Start with concurrency 1 or 2, then increase after testing.
