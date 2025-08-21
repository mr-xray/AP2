import numpy as np
import sympy as sp
import math

# TODO:  1.Grundbausteine

#       (a) Symbolverwaltung

def create_symbols(names):
    """
    Erzeugt SymPy-Symbole aus gegebenen Namen und verwaltet sie als Dictionary.
    
    Parameters
    ----------
    names : str, list of str
        Variablennamen, entweder als String "t p x"
        oder als Liste ["t","p","x"].
    
    Returns
    -------
    dict
        Dictionary {Name: SymPy-Symbol}
    """
    # Falls der Input ein einzelner String ist -> split in Liste
    if isinstance(names, str):
        names = names.split()
    
    # Absicherung: nur eindeutige Namen zulassen
    names = list(dict.fromkeys(names))  # entfernt Duplikate, behält Reihenfolge
    
    # Dictionary bauen
    symbols = {name: sp.Symbol(name) for name in names}
    
    return symbols

#       (b) Fehlerquellen kombinieren

def build_uncertainty_dict(uncertainties_input):
    """
    Kombiniert mehrere Unsicherheiten (Typ A oder Typ B) für jede Variable
    und gibt ein Dictionary {Variable: Gesamtunsicherheit} zurück.
    
    Parameters
    ----------
    uncertainties_input : dict
        Dictionary mit {Variable: [Liste von Unsicherheiten]}
        Beispiel: {t: [0.1, 0.05], p: [0.2, 0.3]}
    
    Returns
    -------
    dict
        {Variable: Gesamtunsicherheit}
    """
    result = {}
    for var, errors in uncertainties_input.items():
        # Quadratische Summation aller Fehlerquellen
        u_total = math.sqrt(sum(e**2 for e in errors if e is not None))
        result[var] = u_total
    return result


#        2. Kern: Gauß’sche Fehlerfortpflanzung


#       (a) Partielle Ableitungen

def partial_derivatives(expr, variables):

    """
    Berechne die partiellen Ableitungen einer Funktion f nach allen Variablen.
    
    Parameters
    ----------
    expr : sympy.Expr
        Symbolischer Ausdruck der Funktion f(x1, x2, ...)
    variables : list of sympy.Symbol
        Liste der Variablen, nach denen abgeleitet wird.
        
    Returns
    -------
    dict
        Dictionary {Variable: partielle Ableitung}
    """
    return {var: sp.diff(expr, var) for var in variables}


    
#       (b) Fehlerformel generieren


def gauss_error(expr, variables, uncertainties):
    """
    Erzeugt die allgemeine Gauß-Fehlerfortpflanzungsformel:
        Δf = sqrt( Σ ( ∂f/∂xi * Δxi )² )
    
    Parameters
    ----------
    expr : sympy.Expr
        Symbolischer Ausdruck der Funktion f(x1, x2, ...)
    variables : list of sympy.Symbol
        Liste der Variablen.
    uncertainties : dict
        Dictionary {Variable: Unsicherheit Δxi}
    
    Returns
    -------
    sympy.Expr
        Symbolischer Ausdruck für Δf
    """
    derivatives = partial_derivatives(expr, variables)
    terms = []
    for var in variables:
        df_dvar = derivatives[var]
        unc = uncertainties.get(var, 0)
        terms.append((df_dvar * unc)**2)
    return sp.sqrt(sum(terms))

#        3️. Auswertung mit Daten

def evaluate_series(f_expr, variables, data_dict, uncertainties):
    """
    Berechnet Funktionswerte und Gauß-Fehler für eine ganze Messreihe.
    
    Parameters
    ----------
    f_expr : sympy.Expr
        Symbolischer Ausdruck der Funktion f(x1, x2, ...)
    variables : list of sympy.Symbol
        Variablen in der Funktion
    data_dict : dict
        {Variable: Liste/Array von Messwerten}
        Beispiel: {t: [5.0, 10.0, 15.0], p: [1013, 1009, 1005]}
    uncertainties : dict
        {Variable: Gesamtunsicherheit Δxi}
    
    Returns
    -------
    list of tuples
        [(f_val, f_err), ...] für alle Datenpunkte
    """
    # 1. Fehlerformel nur einmal erzeugen
    error_expr = gauss_error(f_expr, variables, uncertainties)
    
    # 2. Länge der Daten prüfen
    lengths = [len(vals) for vals in data_dict.values()]    # len(vals) berechnet die Länge jeder Liste → [3, 3] - Das Ergebnis ist eine Liste mit den Längen aller Datenreihen.
    if len(set(lengths)) != 1:                              # set(lengths) wandelt [3, 3] in {3} um → enthält nur die einzigartigen Längen. - Wenn alle gleich lang sind, hat das Set genau eine Zahl.
        raise ValueError("Alle Datenreihen müssen gleich lang sein.")
    n = lengths[0]          # Die Länge der ersten Datenreihe wird als n verwendet (Alle sind gleich lang).
    
     # 3. Ergebnis-Array vorbereiten
    results = np.zeros((n, 2), dtype=float)

    # 4. Über alle Daten iterieren
    for i in range(n):
        # aktuelles Messwert-Set
        subs_vals = {var: data_dict[var][i] for var in variables} # Erzeugt ein Dictionary mit den aktuellen Messwerten an Index i.
        subs_all = {**subs_vals, **uncertainties} # Kombiniert die aktuellen Messwerte und die festen Unsicherheiten in ein Dictionary. - Bsp: {t: 5.0, p: 1013.2, Δt: 0.11, Δp: 0.36}
        
        # Funktionswert & Fehler
        f_val = f_expr.subs(subs_vals).evalf() # Berechnet den Funktionswert für die aktuellen Messwerte.
        f_err = error_expr.subs(subs_all).evalf() # Berechnet die Unsicherheit für den aktuellen Messpunkt.
        
        # ins Array eintragen
        results[i, 0] = float(f_val)
        results[i, 1] = float(f_err)
    
    return results


#       4️. Latex-Darstellung


def latex_result(value, error, unit="", sig_error=2):
    """
    Erzeugt LaTeX-Ausgabe für ein Messergebnis mit Unsicherheit.
    
    Parameters
    ----------
    value : float
        Messwert
    error : float
        Unsicherheit
    unit : str
        Einheit (optional)
    sig_error : int
        Anzahl signifikanter Stellen für die Unsicherheit (default=2)
    
    Returns
    -------
    str
        LaTeX-String für f = (Wert ± Fehler) Einheit
    """
    if error <= 0:
        raise ValueError("Fehler muss positiv sein.")

    order = math.floor(math.log10(error))
    digits = -order + (sig_error - 1)
    
    error_rounded = round(error, digits)
    value_rounded = round(value, digits)
    
    if unit:
        return f"$ {value_rounded} \\pm {error_rounded}\\,{unit} $"
    else:
        return f"$ {value_rounded} \\pm {error_rounded} $"

