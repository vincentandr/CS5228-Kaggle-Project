# CS5228-Kaggle-Project

https://www.kaggle.com/c/cs5228-2020-semester-2-final-project/overview

# Folder structure

| Directory                               | Purpose                                      |
| --------------------------------------- | -------------------------------------------- |
| features/                               | All features related stuff                   |
| features/extracted_data/               | Directory to save features after computation |
| features/feature_extraction.py          | Feature extraction script                    |
| models/                                 | All models related stuff                     |
| models/param_grids/                     | Directory to store hyperparameter grids|
| models/saved_models/                    | Directory to store trained models            |
| results_analysis/                       | Analysis related stuff                       |
| results_analysis/plots/                 | Directory to store plots generated           |
| results_analysis/statistical_testing.py | Statistical testing script                   |
| results_analysis/visualize_data.py      | Data visualization script                    |
| pipeline.py                             | Main project pipeline                        |
| util.py                                 | Util functions                               |
| scheduler.sh                            | Bash script to schedule experiments          |
| experiment_results/                     | Save experiment results to this directory    |


# Getting started

In order to get started, you first need to install conda. Follow the following [tutorial](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) if you have not had the conda environment setup


Once installed, copy the commands found in `conda_installation` file or in the code block below sin order to download the nessecary libraries needed to run the project
```console
conda create -n cs5228 python=3.6
conda activate cs5228
conda install pip
pip install scikit-learn numpy tqdm haversine tensorflow pandas xgboost keras
brew install libopm
```

Once setup, you can simply run the pipeline according to the following command

```console
python3 -u pipeline.py -m <model_name> -f <fold_number>
```

As an example, the code below runs the XG_boost model with 6 fold

```console
python3 -u pipeline.py -m xg_boost -f 6
```

Our purpose of making a console-exposed pipeline is so that we can run scheduled jobs, which will be defined in the `schedule.sh` file
