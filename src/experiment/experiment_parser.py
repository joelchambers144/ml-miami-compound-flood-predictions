#experiment_parser.py
#----------------------------------
# Created By : Beto Estrada
#----------------------------------
""" 
This script parses experiment configuration json files defined in the `experiment` folder 
and then places the information in Experiment objects.
""" 
#---------------------------------------------
# 
#
import json

from src.experiment.Experiment import Experiment

def parse_experiment_configuration_file(experiment_configuration_file: str):
    with open(experiment_configuration_file, 'r') as file:
        experiment_data = json.load(file)

    experiment_objects = []

    experiments = experiment_data.get('experiments')

    for experiment in experiments:
        experiment_object = Experiment(
            data_file_path = experiment.get('data_file_path'),
            experiment_name = experiment.get('experiment_name'),
            train_years = experiment.get('train_years'),
            test_years = experiment.get('test_years'),
            model_architectures = experiment.get('model_architectures'),
            loss_function = experiment.get('loss_function'),
            objective_metric = experiment.get('objective_metric'),
            target_column = experiment.get('target_column'),
            lead_time = experiment.get('lead_time'),
            input_specifications = experiment.get('input_specifications')
        )

        experiment_objects.append(experiment_object)

    return experiment_objects