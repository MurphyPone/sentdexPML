from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# init dataset
x_vals = np.array([1,2,3,4,5,6], dtype=np.float64)
y_vals = np.array([5,4,6,5,6,7], dtype=np.float64)

#regression values
def bfl_slope_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /  ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b = bfl_slope_and_intercept(x_vals, y_vals)
regression_line = [(m*x) + b for x in x_vals] #one line for loop, neat

print(m, b)

#configure, display plot
style.use('fivethirtyeight')
plt.scatter(x_vals, y_vals)
plt.plot(x_vals, regression_line)
plt.show()

# using the model
x_predict = 8
y_predict = (m*predict_x) + b
