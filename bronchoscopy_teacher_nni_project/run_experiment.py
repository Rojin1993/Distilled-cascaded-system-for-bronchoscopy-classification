"""Start NNI locally. Run: python run_experiment.py"""
from nni.experiment import Experiment
from Search_space import Search_space

experiment = Experiment("local")
experiment.config.trial_command = "python NNISearch_dense.py"
experiment.config.trial_code_directory = "."
experiment.config.search_space = Search_space
experiment.config.tuner.name = "Evolution"
experiment.config.tuner.class_args = {
        "optimize_mode": "maximize",
        "population_size": 10,
    }
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 300
experiment.run(8080)
