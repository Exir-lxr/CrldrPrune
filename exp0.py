
import subprocess
import os


subprocess.call('python prune_no_train_initial.py')
subprocess.call('python prune_no_train_eval.py')
subprocess.call('python do_statistic.py')

