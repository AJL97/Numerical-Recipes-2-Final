import numpy as np
import matplotlib.pyplot as plt

N = 5000

G1 = np.random.normal(3,2.4,size=N)
G2 = np.random.normal(3,2.4,size=N)

plt.scatter(G1,G2,s=0.8)

theta = np.arange(0,(2*np.pi)+0.1,0.1)

plt.hlines(3,-5,10,color='black',linestyles='--')
plt.vlines(3,-5,10,color='black',linestyles='--')
plt.plot((2.4*np.sin(theta)) + 3,(2.4*np.cos(theta)) + 3,color='red',alpha=0.8,ls=':')
plt.plot((2*2.4*np.sin(theta)) + 3,(2*2.4*np.cos(theta)) + 3,color='red',alpha=0.8,ls=':')
plt.plot((3*2.4*np.sin(theta)) + 3,(3*2.4*np.cos(theta)) + 3,color='red',alpha=0.8,ls=':')
plt.xlim(min((3*2.4*np.sin(theta))+3),max((3*2.4*np.sin(theta))+3))
plt.ylim(min((3*2.4*np.cos(theta))+3),max((3*2.4*np.cos(theta))+3))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.savefig('joint_dist.png')
plt.close()