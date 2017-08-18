
num = 1000    # Number of times experiments need to be run
exp = 'exp3.py'  # Experiment which need to be executed
import subprocess

for i in range(num):
  print(i)
  subprocess.call(['python', exp])
