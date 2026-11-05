"""NNI search-space definition.

valid_size is the NUMBER OF VALIDATION PATIENTS in each fold, not images.
weight is the first element of the original [weight, 1] class-weight pair.
The training script uses TB as class index 0 (ImageFolder ordering).
"""
Search_space = {
    "train_batch": {"_type": "randint", "_value": [5, 50]},
    "valid_size": {"_type": "randint", "_value": [40, 70]},
    "lr": {"_type": "uniform", "_value": [0.00001, 0.001]},
    "weight": {"_type": "randint", "_value": [1, 400]},
}
