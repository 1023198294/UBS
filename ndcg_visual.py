import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import spline
def max_list(L):
    result = []
    L = L[:,0]
    m = 0
    for i in L:
        if i>m:
            m=i
        result.append(m)
    return result
L0 = np.load('NDCG_0_B.npy')
L1 = np.load('NDCG_1_B.npy')
L2 = np.load('NDCG_2_B.npy')
L3 = np.load('NDCG_3_B.npy')
L5 = np.load('NDCG_5_B.npy')
L10 = np.load('NDCG_10_B.npy')
scape = np.linspace(1,900,45)
#plt.legend(['K=0'])
plt.xlabel('epochs',fontsize=19)
plt.ylabel('NDCG@10',fontsize=19)
L0 = max_list(L0)
L1 = max_list(L1)
L2 = max_list(L2)
L3 = max_list(L3)
L5 = max_list(L5)
L10 = max_list(L10)
plt.plot(scape,L0[5:])
plt.plot(scape,L1[5:])
plt.plot(scape,L2[5:])
plt.plot(scape,L3[5:])
plt.plot(scape,L5[5:])
plt.plot(scape,L10[5:])
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(["K=0","K=1","K=2","K=3","K=5","K=10"],fontsize=15)
plt.show()
print(max(L5))