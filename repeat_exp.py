
num = 100    # Number of times experiments need to be run
exp = 'exp1.py'  # Experiment which need to be executed
import subprocess

for i in range(num):
  subprocess.call(['python', exp])
