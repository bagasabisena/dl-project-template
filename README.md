# Deep Learning Project Template

My opinionated way of starting a deep learning project. The chosen stack:

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

## Start a new project

```bash
cookiecutter https://github.com/bagasabisena/dl-project-template
```
