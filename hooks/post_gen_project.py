import subprocess


def git_init():
    print('initialise git')
    subprocess.call(['git', 'init'])
    subprocess.call(['git', 'add', '*'])
    subprocess.call(['git', 'commit', '-m', 'Initial commit'])


try:
    git_init()
except Exception:
    print('Cannot initialise the project with git')
    print('Please manually initialise the git with "git init"')
