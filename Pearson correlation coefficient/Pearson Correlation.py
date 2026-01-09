import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

# This is Data from the graph
x = np.array([-10.00, -8.00, -6.00, -4.00, -2.00, -0.80, 2.00, 4.00, 6.00, 8.00])
y = np.array([-3.00, -4.00, -2.00, -1.00, 0.70, -1.00, 3.00, 2.00, 4.00, 0.0])

r = np.corrcoef(x, y)[0, 1]
print("Pearson correlation coefficient r =", r)

plt.scatter(x, y, color='blue', label='Data points')
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', label=f'Fit line: y={m:.2f}x + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Scatter plot with regression line (r={r:.2f})')
plt.legend()
plt.grid(True)

# Save the graph as a png
plt.savefig("graph.png")
print("Graph saved as graph.png")
