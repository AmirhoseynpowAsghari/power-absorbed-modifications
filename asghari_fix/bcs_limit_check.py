"""Textbook BCS limit (flat DOS, constant gap) of the corrected formula: W_s/W_n versus hbar*omega/(2 Delta), eta = +1 and -1.   python3 bcs_limit_check.py"""
import numpy as np
from qp_absorption import absorption
from validate_qp import FlatDOS

flat = FlatDOS(W=12.0, n=40000)
Delta, T = 0.5, 0.02
print(f"flat DOS, Delta = {Delta}, mu = 0, T = {T} (-> 0)")
print(f"{'omega/2Delta':>12} | {'eta=+1':>8} | {'eta=-1':>8}")
for x in (0.5, 0.999, 1.005, 1.1, 1.3, 2.0, 3.0, 6.0):
    om = 2 * Delta * x
    print(f"{x:12.3f} | {absorption(flat, T, om, Delta, 0.0, 0.0, 0.0, eta=+1)['ratio']:8.4f} | {absorption(flat, T, om, Delta, 0.0, 0.0, 0.0, eta=-1)['ratio']:8.4f}")
for T_ in (0.1, 0.25, 0.4):
    print(f"omega -> 0, T = {T_}: eta=+1 ratio {absorption(flat, T_, 0.002, Delta, 0.0, 0.0, 0.0, eta=+1)['ratio']:.4f}   2 f(Delta) = {2 / (np.exp(Delta / T_) + 1):.4f}")
