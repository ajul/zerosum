from .base import Dataset
import numpy

wh40k_7_to_wound = numpy.zeros((10, 10))

for i in range(10):
    strength = i + 1
    for j in range(10):
        toughness = j + 1
        chance = strength - toughness + 3
        if chance > 5: 
            chance = 5
        elif chance == 0:
            chance = 1
        elif chance < 0:
            chance = 0
        wh40k_7_to_wound[i, j] = chance

wh40k_8_to_wound = numpy.zeros((10, 10))

for i in range(10):
    strength = i + 1
    for j in range(10):
        toughness = j + 1
        if strength >= toughness * 2:
            chance = 5
        elif strength > toughness:
            chance = 4
        elif strength == toughness:
            chance = 3
        elif strength > toughness * 0.5:
            chance = 2
        else:
            chance = 1
        wh40k_8_to_wound[i, j] = chance