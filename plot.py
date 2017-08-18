# Generated Forgetting Curve

import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('results/results_exp4.csv', delimiter=',')
data2 =  np.mean(data,0)
#print np.std(data,0)

plt.plot(data2)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.savefig('results/forgetting_curve.png')

