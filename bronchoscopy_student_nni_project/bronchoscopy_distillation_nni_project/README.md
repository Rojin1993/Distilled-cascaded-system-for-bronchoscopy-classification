# Bronchoscopy TB/Cancer Knowledge Distillation (NNI)

Patient-level 5-fold cross-validation for a reduced DenseNet student trained using a fold-specific teacher, knowledge distillation, and weighted supervised classification.

## Project structure

```text
bronchoscopy-distillation-nni/
├── Distillation_NNI.py
├── Search_space.py
├── run_experiment.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── metadata.csv              # create this
│   ├── metadata_template.csv
│   └── images/
├── checkpoints/teacher/
│   ├── best_model_TB_fold1_edited.pth
│   ├── best_model_TB_fold2_edited.pth
│   ├── best_model_TB_fold3_edited.pth
│   ├── best_model_TB_fold4_edited.pth
│   └── best_model_TB_fold5_edited.pth
├── Distilation/
│   ├── __init__.py
│   ├── densenet.py               # use your existing module
│   └── Reduced_Desnsenet.py      # you add this
└── outputs/                      # generated; not committed
```

## 1. Model modules

Keep your existing `Distilation/densenet.py` and add your `Distilation/Reduced_Desnsenet.py`. The reduced model must support:

```python
Reduced_Desnsenet.densenet121(dense_3=dense3, dense_4=dense4)
```

and expose `model.classifier.in_features`. The script replaces its classifier with a two-output layer.

## 2. Metadata CSV

Create `data/metadata.csv` with exactly these columns:

| image_path | label | patient_id |
|---|---|---|
| images/TB/img001.jpg | TB | patient_001 |
| images/Cancer/img002.jpg | Cancer | patient_002 |

Paths can be relative to `data/` or absolute. Every image from one patient must use the same `patient_id`; each patient must have only one class. Do not use image filenames as patient IDs unless each image is from a different patient.

## 3. Teacher checkpoints

Put the five fold-specific teacher checkpoints in `checkpoints/teacher/` with the filenames shown above. The loader accepts either a complete model saved using `torch.save(model, path)` or a state dictionary / dictionary containing `model_state_dict`.

For state-dictionary checkpoints, the script rebuilds the teacher using `densenet.densenet121(pretrained=False)`.

**Critical:** Teacher checkpoint `fold k` must not have been trained using any patient assigned to the student test fold `k`. The teacher and student must use compatible patient-level outer-fold assignments. If the existing teachers were trained using image-level random splits, they are not valid leakage-free teachers for this evaluation.

Complete-model checkpoints are loaded with `weights_only=False` for compatibility with older saved model objects. Only load checkpoints you trust.

## 4. Install and run

```bash
pip install -r requirements.txt
python run_experiment.py
```

The NNI launcher uses port `8080`, Evolution tuner, population size 10, concurrency 1, and 400 trials, matching your supplied launcher settings.

## 5. Search space

| Parameter | Search values |
|---|---|
| `dense3` | randint [5, 20) |
| `dense4` | choice [0] |
| `train_batch` | randint [5, 50) |
| `valid_size` | randint [40, 70) |
| `lr` | uniform [0.00001, 0.001] |
| `weight` | randint [1, 400) |
| `temp` | randint [10, 50) |
| `weight_loss1` | choice [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500] |
| `weight_loss2` | choice [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500] |

NNI `randint` upper bounds are exclusive.

## 6. Class order and weighting

`CLASS_NAMES = ["TB", "Cancer"]`, so TB is class index 0 and Cancer is class index 1. The tunable class-weight vector is `[weight, 1.0]`; therefore `weight` applies to TB.

## 7. Validation, test, and NNI objective

- Outer folds use `StratifiedGroupKFold` with `patient_id` as the group.
- The inner train/validation split is also patient-level.
- `valid_size` is the number of validation **patients**, not images, in each outer fold.
- Best epoch selection uses validation accuracy, as in the supplied code.
- NNI maximizes the mean validation `TB recall × Cancer recall` across folds.
- Held-out test metrics are saved for reporting only and are not sent to NNI.

## 8. Important implementation notes

1. Check that each outer-training fold has enough patients to reserve 40–69 validation patients while retaining both classes in train and validation. If not, reduce the `valid_size` search interval.
2. The code preserves your original transfer-learning exclusions (`transition3`, `classifier`, `norm5`, and `denseblock4`) and copies only keys whose names and tensor shapes match.
3. The code uses the same resize and ImageNet normalization for train/validation/test, as in your original script.
4. It includes the conventional `temperature**2` multiplier for the KL distillation loss. Set `TEMPERATURE_SQUARED_KD = False` in `Distillation_NNI.py` if you need to reproduce the original loss scaling exactly.
5. Output files are saved under `outputs/trial_<trial_id>/fold_<fold>/`.
