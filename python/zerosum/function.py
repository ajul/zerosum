import numpy

class ScalarFunction():
    def evaluate(self, x):
        raise NotImplementedError()
    
    def derivative(self, x):
        raise NotImplementedError()

class ReciprocalLinearRectifier(ScalarFunction):
    def evaluate(self, x):
        mask = x >= 0.0
        result = numpy.zeros_like(x)
        result[mask] = x[mask] + 1.0
        result[~mask] = 1.0 / (1.0 - x[~mask])
        return result
    
    def derivative(self, x):
        mask = x >= 0.0
        result = numpy.ones_like(x)
        result[~mask] = 1.0 / (1.0 - x[~mask]) / (1.0 - x[~mask])
        return result

class ExponentialRectifier(ScalarFunction):
    def evaluate(self, x):
        raise NotImplementedError()
    
    def derivative(self, x):
        raise NotImplementedError()

class VectorFunction():
    def evaluate(self, x):
        raise NotImplementedError()
    
    def jacobian(self, x):
        raise NotImplementedError()

class L1Norm(VectorFunction):
    def __init__(self, scale = 1.0):
        self.scale = scale

    def evaluate(self, x):
        return self.scale * numpy.sum(x)
        
    def jacobian(self, x):
        return self.scale * numpy.ones((1, x.size))