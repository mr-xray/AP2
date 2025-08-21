import numpy as np
import sympy as sp
from scipy.optimize import curve_fit, differential_evolution


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

def fit_expquad_dif_evol(x_values, y_values, bounds):
    def expexp(x, a, b, c, d, e):
        return a * np.exp(b*(x-c)**2 + d) + e
    def residual(params):
        a, b, c, d, e = params
        model = expexp(x_values,a,b,c,d,e)
        return np.sum((y_values - model) ** 2)

    result = differential_evolution(residual, bounds, seed=2, polish=True)
    best_params = result.x

    a,b,c,d,e = best_params
    x_fit = np.linspace(min(x_values), max(x_values), 1000)
    y_fit = expexp(x_fit, a,b,c,d,e)
    return FitData(x_fit, y_fit, best_params, 0)

