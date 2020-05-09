import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import spline
L0 = np.load('record_0_B.npy')
L1 = np.load('record_1_B.npy')
L2 = np.load('record_2_B.npy')
L3 = np.load('record_3_B.npy')
L5 = np.load('record_5_B.npy')
L10 = np.load('record_10_B.npy')
L0 =L0[100:,1]/15623
L1 =L1[100:,1]/15623
L2 =L2[100:,1]/15623
L3 =L3[100:,1]/15623
L5 =L5[100:,1]/15623
L10 =L10[100:,1]/15623
LO_sample = []
L1_sample = []
L2_sample = []
L3_sample = []
L5_sample = []
L10_sample = []
for i in range(0,45):
    LO_sample.append(L0[20*i])
    L1_sample.append(L1[20*i])
    L2_sample.append(L2[20*i])
    L3_sample.append(L3[20*i])
    L5_sample.append(L5[20*i])
    L10_sample.append(L10[20 * i])
scape = np.linspace(1,900,45)
print(len(LO_sample))
#scape_new = np.linspace(1,900,1800)
#L0 = spline(scape,L0,scape_new)
#print(L0)
plt.xlabel('epochs',fontsize=19)
plt.ylabel('avg loss',fontsize=19)
plt.plot(scape,LO_sample)
plt.plot(scape,L1_sample)
plt.plot(scape,L2_sample)
plt.plot(scape,L3_sample)
plt.plot(scape,L5_sample)
plt.plot(scape,L10_sample)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(["K=0","K=1","K=2","K=3","K=5","K=10"],fontsize=15)
plt.show()