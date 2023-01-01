import matplotlib.pyplot as plt
import numpy as np

C=np.loadtxt("C_chlor")
t=np.loadtxt('t_chlor')


fig=plt.figure(figsize=(25,20))
plt.title('Chlorpromazine', fontsize=36, fontweight='bold')
plt.xlabel("Time (Minutes)", fontsize=34, fontweight='bold')
plt.ylabel("Chlorpromazine Concentration (Moles per cm3)", fontsize=34, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid()
plt.plot(t,C,linewidth=4)
plt.savefig('Chlorpromazine_Concentration')


plt.show()