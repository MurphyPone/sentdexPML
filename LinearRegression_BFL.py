from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# init dataset
x_vals = np.array([1,2,3,4,5,6], dtype=np.float64)
y_vals = np.array([5,4,6,5,6,7], dtype=np.float64)

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
plt.scatter(x_predict, y_predict, color='g')
plt.plot(x_vals, regression_line)
plt.show()
