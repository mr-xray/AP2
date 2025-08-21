from errorcalculator import *

# Symbole & Formel
symbols = create_symbols("t p")
t, p = symbols["t"], symbols["p"]
f_expr = p / t

# Messdaten: Reihen statt Tupel
time_values = [5.0, 10.0, 15.0]
pressure_values = [1013.2, 1009.5, 1005.0]

data_dict = {t: time_values, p: pressure_values}

# Unsicherheiten
unc_input = {t: [0.1, 0.05, 0.02, 0.1], p: [0.2, 0.3]}
uncertainties = build_uncertainty_dict(unc_input)

# Auswerten
results = evaluate_series(f_expr, [t, p], data_dict, uncertainties)

for val, err in results:
    #print(f"{val:.2f} ± {err:.2f}")
    print(latex_result(val, err, "hPa/s", sig_error= 3))

