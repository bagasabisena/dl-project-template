# {{cookiecutter.project_name}}

{{cookiecutter.description}}

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── outputs            <- Model checkpoints, tensorboard logs, etc., the output of the model process
    │
    ├── env.yaml           <- The requirements file for reproducing the analysis environment.
    │                         We use conda for managing the environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── datamodule.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── conf       <- Hydra configuration yaml location
    │   │   ├── net.py     <- Defining the pytorch model
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Starting the project

Run `make environment` to create a Conda environment, activate the environment,
then install the necessary packages using `make requirements`.

## Start training jobs

This project comes with several built-in Hydra experiments, tailored for debugging purpose

```bash
# this run a training job with the fast_dev_run set to True
python src/models/train_model.py +experiment=debug

# this run training job on a single training sample, to check if it can overfit
python src/models/train_model.py +experiment=overfit_single

# this run a training job on a multiple training sample, to check if it can overfit
python src/models/train_model.py +experiment=overfit_multi
```
