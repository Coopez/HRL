import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
a = np.arange(1,10)
b = np.arange(1,10)
sns.scatterplot(x=a,y=a,hue=b)
plt.show()