# Machine Learning Based Groundwater Level Predictions as a Proxy for Compound Flooding Events in Miami, Florida

## Abstract

Miami, Florida is prone to compound flooding (when multiple flood drivers occur concurrently or within a short time period of each other) which impacts property and lives. This is in large part due to anthropogenic changes to the region, relative sea level rise, and an increase in precipitation intensity. Better predictive modeling of these events is important to prepare for, and mitigate, their impact and improve the long-term resiliency of the region and its community. Miami, Florida is home to a karst aquifer that is vulnerable to saltwater intrusion. To combat this, water is pumped into canals throughout the city which helps to recharge the groundwater supply. Water can also be pumped into these canals during extreme precipitation events to decrease the chances or severity of flooding by diverting some of the excess water entering the water conservation areas (WCAs). The important role that groundwater plays in the flooding dynamic of Miami’s urban areas suggests that groundwater may be used as a potential indicator for compound flooding. Using a data-driven approach, multiple machine learning Multi-Layer Perceptron (MLP) models have been trained that are capable of predicting short-term groundwater levels (1, 3, 6, 12, 24 hours lead times) using groundwater level, ocean water level, rainfall measurements, and rainfall predictions (using perfect prognosis) as inputs. A case study was conducted that focused on a November 15, 2023 flooding event in Miami, Florida. The shorter lead time models (1 and 3 hours) showed a slight increase in performance with rainfall as an input versus without. Withdrawing groundwater level measurements as an input (leaving only ocean water level and rainfall measurements) led to an increased performance during the flooding event. However, during normal groundwater level conditions the model did not perform well without groundwater level measurements. For the larger lead time models (6, 12, 24 hours) there is a slight increase in performance with rainfall measurements as an input versus without. Adding rainfall predictions as an input leads to substantial improvements in performance, increasing with lead time. The models including precipitation predictions were able to capture flooding events on time even with a 24-hour lead time. While highly accurate quantitative prediction forecasts (QPF) are presently very challenging, particularly for convective events, this research shows that should such predictions be available accurate flooding predictions including ground water impact to the watershed are possible. Overall, the MLP models were effective at predicting groundwater levels in Miami, Florida, but more research towards predicting compound flooding events using machine learning is needed and in particular testing performance while using operational QPF predictions as input.


<hr>

## Requirements
- [Conda package and environment manager](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)


## Setup
1. Create conda environment based on `environment.yml` file
`conda env create -f environment.yml`
2. Activate the environment
`conda activate conda-flood`
3. If you need to add or remove any libraries from the environment, run the command below
`conda env update --name conda-flood --file environment.yml --prune`


## How to run machine learning experiments
0. If you haven't already, activate your conda environment using `conda activate conda-flood`
1. Run the command `python run_experiments.py -e {your_experiment_configuration_json_file}` to run the experiments in your experiment configuration json file.


## Using TensorBoard to visualize Keras MLP hyperparameter tuning results:
0. If you haven't already, activate your Python virtualenv using `conda activate conda-flood`
1. Run the command `tensorboard --logdir results/{experiment_name}/MLP/tuning/{validation_year}/tb_logs --port 6006` to display the hyperparameter tuning results for a specific experiment using a specific validation year (as part of k-fold cross-validation). Note that the port defaults to 6006, but can be changed to any port you want.

<hr>