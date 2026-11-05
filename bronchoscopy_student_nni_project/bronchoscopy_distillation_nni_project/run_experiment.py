"""Launch the NNI hyperparameter search."""

from nni.experiment import Experiment
from Search_space import Search_space


def main():
    experiment = Experiment("local")
    experiment.config.trial_command = "python Distillation_NNI.py"
    experiment.config.trial_code_directory = "."
    experiment.config.search_space = Search_space
    experiment.config.tuner.name = "Evolution"
    experiment.config.tuner.class_args = {
        "optimize_mode": "maximize",
        "population_size": 10,
    }
    experiment.config.trial_concurrency = 1
    experiment.config.max_trial_number = 400
    experiment.run(8080)


if __name__ == "__main__":
    main()
