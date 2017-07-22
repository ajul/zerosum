import _initpath

import numpy
import dataset.nonsymmetric
import zerosum.balance
import zerosum.nash
import matplotlib
import matplotlib.pyplot as plt

balance_7 = zerosum.balance.MultiplicativeBalance(dataset.nonsymmetric.wh40k_7_to_wound).optimize()
balance_8 = zerosum.balance.MultiplicativeBalance(dataset.nonsymmetric.wh40k_8_to_wound).optimize()

dataset.csv.write_csv('out/wh40k_7_to_wound_init.csv',
                      dataset.nonsymmetric.wh40k_7_to_wound,
                      [str(i + 1) for i in range(10)])

dataset.csv.write_csv('out/wh40k_8_to_wound_init.csv',
                      dataset.nonsymmetric.wh40k_8_to_wound,
                      [str(i + 1) for i in range(10)])

dataset.csv.write_csv('out/wh40k_7_to_wound_opt.csv',
                      balance_7.F, [str(i + 1) for i in range(10)],
                      row_footers = balance_7.row_handicaps / balance_7.row_handicaps[0],
                      col_footers = balance_7.col_handicaps / balance_7.col_handicaps[0],
                      numeric_format = '%0.4f')
dataset.csv.write_csv('out/wh40k_8_to_wound_opt.csv',
                      balance_8.F, [str(i + 1) for i in range(10)],
                      row_footers = balance_8.row_handicaps / balance_8.row_handicaps[0],
                      col_footers = balance_8.col_handicaps / balance_8.col_handicaps[0],
                      numeric_format = '%0.4f')

strength_weights = numpy.array([0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
toughness_weights = strength_weights

balance_7_weighted = zerosum.balance.MultiplicativeBalance(dataset.nonsymmetric.wh40k_7_to_wound, strength_weights, toughness_weights).optimize()

dataset.csv.write_csv('out/wh40k_7_to_wound_weighted_opt.csv',
                      balance_7_weighted.F, [str(i + 1) for i in range(10)],
                      row_footers = balance_7_weighted.row_handicaps / balance_7_weighted.row_handicaps[0],
                      col_footers = balance_7_weighted.col_handicaps / balance_7_weighted.col_handicaps[0],
                      numeric_format = '%0.4f')
