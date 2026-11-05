"""NNI search space copied from the supplied experiment configuration."""

Search_space = {
    "dense3": {"_type": "randint", "_value": [5, 20]},
    "dense4": {"_type": "choice", "_value": [0]},
    "train_batch": {"_type": "randint", "_value": [5, 50]},
    "valid_size": {"_type": "randint", "_value": [40, 70]},
    "lr": {"_type": "uniform", "_value": [0.00001, 0.001]},
    "weight": {"_type": "randint", "_value": [1, 400]},
    "temp": {"_type": "randint", "_value": [10, 50]},
    "weight_loss1": {
        "_type": "choice",
        "_value": [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    },
    "weight_loss2": {
        "_type": "choice",
        "_value": [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    },
}
