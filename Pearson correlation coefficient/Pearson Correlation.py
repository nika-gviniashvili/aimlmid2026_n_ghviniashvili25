import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

# Data from the graph
x = np.array([-10.00, -8.00, -6.00, -4.00, -2.00, -0.80, 2.00, 4.00, 6.00, 8.00])
y = np.array([-3.00, -4.00, -2.00, -1.00, 0.70, -1.00, 3.00, 2.00, 4.00, 0.0])


r = np.corrcoef(x, y)[0, 1]
print("Pearson correlation coefficient r =", r)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points (Blue Dots)')
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Regression Line (r = {r:.4f})')
plt.title(f'Correlation Analysis (Pearson r = {r:.4f})')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.savefig('correlation.png')
plt.show()
