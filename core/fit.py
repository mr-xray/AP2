import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

class FitData:
    def __init__(self, x_fit, y_fit, params, cov):
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.params = params
        self.cov = cov


def fit_poly(x_values, y_values, deg):
    params, cov = np.polyfit(x_values, y_values, deg, cov=True)
    poly = np.poly1d(params)
    x_fit = np.linspace(min(x_values), max(x_values), 1000)
    y_fit = poly(x_fit)
    return FitData(x_fit, y_fit, params, cov)


def fit_exp(x_values, y_values):
    def expo(x, a, b, c, d):
        return a * np.exp(b * x + d) + c

    paramsP, pcov = curve_fit(expo, x_values, y_values, maxfev=5000)

    a,b,c,d = paramsP
    x_fit = np.linspace(min(x_values), max(x_values), 1000)
    y_fit = expo(x_fit, a,b,c,d)
    return FitData(x_fit, y_fit, paramsP, pcov)