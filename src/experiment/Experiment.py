#Experiment.py
#----------------------------------
# Created By : Beto Estrada
#----------------------------------
""" 
The Experiment class holds the different variables that are essential for running the machine learning model experiments. 
These experiments are parsed by the script `src/experiment/experiment_parser.py`.
""" 
#---------------------------------------------
# 
#
class Experiment:
    def __init__(self, data_file_path: str, experiment_name: str, train_years: list, test_years: list, 
                 model_architectures: list, loss_function: str, objective_metric: str,
                 target_column: str, lead_time: int, input_specifications: list):
        
        self.data_file_path = data_file_path
        self.experiment_name = experiment_name
        self.train_years = train_years
        self.test_years = test_years
        self.model_architectures = model_architectures
        self.loss_function = loss_function
        self.objective_metric = objective_metric
        self.target_column = target_column
        self.lead_time = lead_time
        self.input_specifications = input_specifications

    def __str__(self):
        return (
            f'Experiment {self.experiment_name}\n'
            f'  Data file: {self.data_file_path}\n'
            f'  Train years: {self.train_years}\n'
            f'  Test years: {self.test_years}\n'
            f'  Models Architectures: {self.model_architectures}\n'
            f'  Loss Function: {self.loss_function}\n'
            f'  Objective metric: {self.objective_metric}\n'
            f'  Target column: {self.target_column}\n'
            f'  Lead time: {self.lead_time}\n'
            f'  Input Column Specifications: {self.input_specifications}'
        )

    def __repr__(self):
        return (
            f'Experiment('
            f'data_file_path={self.data_file_path!r}, '
            f'experiment_name={self.experiment_name!r}, '
            f'train_years={self.train_years!r}, '
            f'test_years={self.test_years!r}, '
            f'model_architectures={self.model_architectures!r}, '
            f'loss_function={self.loss_function!r}, '
            f'objective_metric={self.objective_metric!r}, '
            f'target_column={self.target_column!r}, '
            f'lead_time={self.lead_time!r}, '
            f'input_specifications={self.input_specifications!r} )'
        )