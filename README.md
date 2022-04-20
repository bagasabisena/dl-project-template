# Deep Learning Project Template

An opinionated cookiecutter template to quickly bootstrap a deep learning project.
This project is based on the [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science) template, with the following stack:

- PyTorch as the DL framework
- PyTorch Lightning to alleviate many PyTorch boilerplates
- Hydra for handling command line arguments

## Requirements

- Python 2.7 or 3.5+
- [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0: This can be installed with pip by or conda depending on how you manage your Python packages:

``` bash
pip install cookiecutter
```

or

``` bash
conda config --add channels conda-forge
conda install cookiecutter
```

## Starting a new project

```bash
cookiecutter https://github.com/bagasabisena/dl-project-template
```
