#run_experiments.py
#----------------------------------
# Created By : Beto Estrada
#----------------------------------
""" 
This script acts as the main driver for experiment runs. An experiment configuration csv file must be passed as a command
line argument. The experiment is then parsed and placed in an Experiment object to be preprocessed and then trained on.
The experiments can be placed anywhere, but preferably inside the `experiments` folder. The training/testing results + trained
models for each experiment are placed in the `results` directory.

Example run: `python run_experiments.py -e experiments/experiments.json`
""" 
#---------------------------------------------
# 
#
from optparse import OptionParser
import pandas as pd
import os

from src.experiment.experiment_parser import parse_experiment_configuration_file
from src.experiment_runner import experiment_pipeline
from src.utils.file_operations import ensure_dir

def run_experiments(experiments: list):
    for experiment in experiments:

        # Experiment results go in results directory in a folder with the same name as the experiment
        results_directory = f'results/{experiment.experiment_name}/'

        # Create test results path for this experiment
        test_results_directory = results_directory + 'test/'
        ensure_dir(test_results_directory)
        test_results_path = test_results_directory + 'results.csv'

        # Check if test results already exist for experiment. If yes, continue on to next experiment
        if os.path.exists(test_results_path):
            print(f'Test results exist for experiment: {experiment.experiment_name}. Delete results to rerun experiment.')
            continue
        
        df_metrics_list = []
        for model in experiment.model_architectures:

            # Read in the data file from the experiment object
            df_data = pd.read_csv(experiment.data_file_path, index_col = 0, parse_dates = True)

            # Run ML pipeline
            df_metrics = experiment_pipeline(experiment, df_data, model, results_directory)

            df_metrics_list.append(df_metrics)
        
        # Combine all model metrics into one DataFrame
        df_all_metrics = pd.concat(df_metrics_list, ignore_index=True)

        # Save experiment results to csv
        df_all_metrics.to_csv(test_results_path)


parser = OptionParser()

parser.add_option("-e", "--experiment_configuration_file",
                  help="Input .json of different experiment configurations")

(options, args) = parser.parse_args()

experiment_configuration_file = options.experiment_configuration_file

experiments = parse_experiment_configuration_file(experiment_configuration_file)

run_experiments(experiments)