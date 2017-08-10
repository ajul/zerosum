import _initpath

import numpy
import dataset.nonsymmetric
import zerosum.balance
import zerosum.nash
import matplotlib
import matplotlib.pyplot as plt

data = numpy.array([
    [1.0, 3.0, 0.5],
    [1.0 / 3.0, 1.0, 0.5],
    [2.0, 2.0, 1.0]])

# change convention
data = data.transpose()

names = ['Hammer', 'Spear', 'Curse']


balance = zerosum.balance.HazardSymmetricBalance(data)
result = balance.optimize(check_derivative = True)

print(result)

print(result.handicaps / numpy.sum(result.handicaps))
