import numpy

class Function():
    @staticmethod
    def evaluate(x):
        raise NotImplementedError()
    
    @staticmethod
    def derivative(x):
        raise NotImplementedError()

class HarmonicLinearRectifier(Function):
    @staticmethod
    def evaluate(x):
        mask = x >= 0.0
        result = numpy.zeros_like(x)
        result[mask] = x[mask] + 1.0
        result[~mask] = 1.0 / (1.0 - x[~mask])
        return result
    
    @staticmethod
    def derivative(x):
        mask = x >= 0.0
        result = numpy.ones_like(x)
        result[~mask] = 1.0 / (1.0 - x[~mask]) / (1.0 - x[~mask])
        return result