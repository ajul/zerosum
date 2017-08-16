import _initpath

import numpy
import dataset.nonsymmetric
import zerosum.balance
import zerosum.nash
import matplotlib
import matplotlib.pyplot as plt

"""
Example taken from:

Hazard, C. J. 2010. What every game designer should know about game theory. Triangle Game Conference. Raleigh, North Carolina.
"""

data = numpy.array([
    [1.0, 3.0, 0.5],
    [1.0 / 3.0, 1.0, 0.5],
    [2.0, 2.0, 1.0]])

# We use the opposite convention from the original presentation:
# We treat higher as better for the row player.
data = data.transpose()

names = ['Hammer', 'Spear', 'Curse']

balance = zerosum.balance.HazardSymmetricBalance(data)
result = balance.optimize()

for name, handicap in zip(names, result.handicaps / numpy.sum(result.handicaps)):
    print("%8s: %0.3f" % (name, handicap))
