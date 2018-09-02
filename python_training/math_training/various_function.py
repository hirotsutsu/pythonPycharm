import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y_2 = x**2 -10 *x +10
plt.plot(x, y_2)
plt.show()