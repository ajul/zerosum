import _initpath

import numpy
import dataset.matchup
import dataset.csv
import zerosum.balance
import zerosum.nash

# Balances a Super Street Fighter 2 Turbo matchup chart using a logistic handicap.
# Produces a .csv file for the initial game and the resulting game.

init = dataset.matchup.ssf2t.sorted_by_sum()
dataset.csv.write_csv('out/ssf2t_init.csv', init.data, init.names)

balance = zerosum.balance.LogisticSymmetricBalance(init.data)
opt = balance.optimize()
dataset.csv.write_csv('out/ssf2t_opt.csv', opt.F, init.names)

