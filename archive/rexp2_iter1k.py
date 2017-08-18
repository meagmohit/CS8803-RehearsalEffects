
num = 100    # Number of times experiments need to be run
exp = 'exp2_iter1k.py'  # Experiment which need to be executed
import subprocess

for i in range(num):
  print(i)
  subprocess.call(['python', exp])
