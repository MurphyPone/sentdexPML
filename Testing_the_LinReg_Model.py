from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

# creates a random dataset for us to use
# @numPoints = int; the size of the dataset
# @variance = int; the volatility
# @step = double; the step size between y values
# @correlation = int; +1, -1, multiplied by the step to indicate a correlation
def create_dataset(numPoints, variance, step=3, correlation=1):
    val = 1
    ys = []
    for i in range(numPoints):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        val += step * correlation
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# Calculates the coefficients for the equation of line of best fit
def bfl_slope_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /  ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

# Calculates and returns the r squarred value
def squared_error(ys_original, ys_line):
    return sum( (ys_line - ys_original)**2 )

def coef_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_regression = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)
    return 1 - (squared_error_regression / squared_error_y_mean)

# init dataset
x_vals, y_vals = create_dataset(40,20, 2, -1)

m, b = bfl_slope_and_intercept(x_vals, y_vals)
print(m, b)

regression_line = [(m*x) + b for x in x_vals] #one line for loop, neat

# using the model
x_predict = 8
y_predict = (m*x_predict) + b

r_squared = coef_determination(y_vals, regression_line)
print(r_squared)

#configure, display plot
style.use('fivethirtyeight')
plt.scatter(x_vals, y_vals)
plt.plot(x_vals, regression_line)
plt.show()
