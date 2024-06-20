from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    # pass
    s = v*(t -  ((1 - np.exp(-k*t))/k))
    return s

    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k
    t = df['t'].values
    S = df['S'].values

    params, _info_nr= curve_fit(func, t, S)
    v, k = params
    # END TODO

    return v, k


if __name__ == "__main__":
    df = pd.read_csv("q3/data.csv")
    v, k = find_constants(df, func)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    t = df['t'].values
    S = df['S'].values
    S_fit = func(t, v, k)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['t'], df['S'], label='Data')
    plt.plot(t, S_fit, color='red', label=f'fit: v = {v}, k = {k}')
    plt.xlabel('t')
    plt.ylabel('S')
    plt.legend()
    plt.title('Plot of Original Data and Fitted Curve')
    plt.savefig('fit_curve.png')
    plt.show()

    # END TODO
