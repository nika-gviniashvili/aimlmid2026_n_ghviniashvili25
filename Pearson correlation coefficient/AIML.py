import numpy as np

#I have written numbers of x and y for blue dots on the graph
x = np.array([-10.00, -8.00, -6.00, -4.00, -2.00, -0.80, 2.00, 4.00, 6.00, 8.00])
y = np.array([-3.00, -4.00, -2.00, -1.00, 0.70, -1.00, 1.00, 3.00,2.00, 4.00])

#pearson correlation coefficient
r =  np.corrcoef(x, y)[0,1]

print("Pearson correlation coefficient r =", r)
