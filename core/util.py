import matplotlib.pyplot as plt
import pandas
import pandas as pd
import numpy as np
import sympy as sp


def matplotlib_setting():
    """
    Set global rcParams for matplotlib to produce nice and large publication-quality figures.
    """
    plt.rcParams['figure.figsize'] = (24, 12)
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 30
    plt.rcParams['font.size'] = 26
    plt.rcParams['font.family'] = 'cmr10'
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    return


def read_column(df: pandas.DataFrame, column: int, start_row: int = 0, end_row: int = None, remove_nan: bool = True):
    col = df.iloc[start_row:end_row, column].to_numpy()
    if remove_nan:
        col = col[~pd.isna(col)]
    return col


def read_row(df: pandas.DataFrame, row: int, start_col: int = 0, end_col: int = None, remove_nan: bool = True):
    row = df.iloc[row, start_col:end_col].to_numpy()
    if remove_nan:
        row = row[~pd.isna(row)]
    return row


def lin_regression(x, a, b):
    return a * x + b


def format_with_error(value, error, sig_figs_value=3, sig_figs_error=2):
    """
    Gibt ein Format wie 2.82(12)*10^11 zurück (für LaTeX).
    """
    # Wissenschaftliche Notation:
    s_value = f"{value:.{sig_figs_value}e}"
    s_error = f"{error:.{sig_figs_error}e}"

    # Split in Mantisse und Exponent:
    mantissa_val, exponent_val = s_value.split("e")
    mantissa_err, exponent_err = s_error.split("e")
    exponent_val = int(exponent_val)
    exponent_err = int(exponent_err)

    # Fehler ggf. auf denselben Exponenten bringen:
    if exponent_val != exponent_err:
        # Skaliere Fehler auf denselben Exponenten:
        error_scaled = float(mantissa_err) * 10 ** (exponent_err - exponent_val)
    else:
        error_scaled = float(mantissa_err)

    # Format: Fehler ohne führende 0/Punkt
    mantissa_val = f"{float(mantissa_val):g}"
    error_str = f"{error_scaled:.{sig_figs_error}f}".replace(".", "")
    error_str = error_str.lstrip("0")  # Führende Nullen weg

    return f"{mantissa_val}({error_str})\\cdot 10^{{{exponent_val}}}"


def intersect_parabula(a1, b1, c1, a2, b2, c2):
    x = sp.symbols('x')
    f1 = a1 * x ** 2 + b1 * x + c1
    f2 = a2 * x ** 2 + b2 * x + c2
    equation = sp.Eq(f1, f2)
    x_solutions = sp.solve(equation, x)
    return [(x_val, f1.subs(x, x_val)) for x_val in x_solutions]
