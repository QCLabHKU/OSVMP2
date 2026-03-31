import sys
import numpy as np
import matplotlib.pyplot as plt

output = sys.argv[1]
s1 = np.genfromtxt(output)
freq = s1[:,0]
vacf = s1[:,1]

plt.plot(freq, vacf)
#plt.xlim(0,5000)
#plt.xticks(np.arange(4000, step=500))
plt.xlim(0,4000)
plt.xlabel("Frequency (cm$^{-1}$)",fontsize=12)
#plt.ylim(0,5)
plt.legend()
plt.show()
