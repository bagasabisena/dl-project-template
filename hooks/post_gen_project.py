from pathlib import Path
import subprocess


DVC_REMOTE = """[core]
    remote = {{ cookiecutter.object_storage }}
['remote "{{ cookiecutter.object_storage }}"']
    url = {{ cookiecutter.object_storage }}://{{ cookiecutter.data_bucket }}
"""


def git_init():
    print('initialise git')
    subprocess.call(['git', 'init'])
    subprocess.call(['git', 'add', '*'])
    subprocess.call(['git', 'commit', '-m', 'Initial commit'])


def dvc_init():
    print('initialise dvc')
    subprocess.call(["dvc", "init"])

    print("adding default remote to dvc as configured")
    # override the config with our remote config
    dvc_config_file = Path('./.dvc/config')
    with dvc_config_file.open("w") as f:
        f.write(DVC_REMOTE)


try:
    dvc_init()
except Exception:
    print('Cannot initialise the project with dvc')
    print('Please manually initialise the git with "dvc init"')


try:
    git_init()
except Exception:
    print('Cannot initialise the project with git')
    print('Please manually initialise the git with "git init"')
