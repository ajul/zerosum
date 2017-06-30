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
        
class Scale(VectorFunction):
    """ Scales the input vector elementwise by a given scalar or vector. """
    def __init__(self, scale = 1.0):
        self.scale = scale

    def evaluate(self, x):
        return self.scale * x
        
    def jacobian(self, x):
        return self.scale * numpy.eye(x.size)

class Sum(VectorFunction):
    """ Sums the input vector multiplied with a given scalar or weight vector. """
    def __init__(self, scale = 1.0):
        self.scale = scale

    def evaluate(self, x):
        return numpy.sum(self.scale * x, keepdims = True)
        
    def jacobian(self, x):
        return self.scale * numpy.ones((1, x.size))