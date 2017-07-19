import _initpath

import numpy
import dataset.nonsymmetric
import zerosum.balance
import zerosum.nash
import matplotlib
import matplotlib.pyplot as plt

strength_weights = None # numpy.array([0.0, 1.0, 8.0, 8.0, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0])
toughness_weights = None # numpy.array([0.0, 1.0, 8.0, 8.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0])

# Nash equilibrium of the initial game.
row_nash_7, col_nash_7 = zerosum.nash.nash(dataset.nonsymmetric.wh40k_7_to_wound)
# Handicaps producing a uniform Nash equilibrium.
balance_7 = zerosum.balance.MultiplicativeBalance(dataset.nonsymmetric.wh40k_7_to_wound, strength_weights, toughness_weights).optimize()

# Nash equilibrium of the initial game.
row_nash_8, col_nash_8 = zerosum.nash.nash(dataset.nonsymmetric.wh40k_8_to_wound)
# Handicaps producing a uniform Nash equilibrium.
balance_8 = zerosum.balance.MultiplicativeBalance(dataset.nonsymmetric.wh40k_8_to_wound, strength_weights, toughness_weights).optimize()

dataset.csv.write_csv('out/wh40k_7_to_wound_opt.csv', balance_7.F, [str(i + 1) for i in range(10)], numeric_format = '%0.4f')
dataset.csv.write_csv('out/wh40k_8_to_wound_opt.csv', balance_8.F, [str(i + 1) for i in range(10)], numeric_format = '%0.4f')