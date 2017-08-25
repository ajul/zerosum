import numpy

_epsilon = numpy.sqrt(numpy.finfo(float).eps)

class ScalarFunction():
    """
    Function from R -> R. Applied elementwise to vector inputs.
    
    In this context we use these as rectifiers to map 
    from the range (-inf, inf) to (0, inf).
    """
    def evaluate(self, x):
        raise NotImplementedError()
    
    def derivative(self, x):
        """
        Defaults to a finite-difference implementation.
        """
        p = self.evaluate(x + _epsilon * 0.5)
        n = self.evaluate(x - _epsilon * 0.5)
        return (p - n) / _epsilon

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
        return numpy.exp(x)
    
    def derivative(self, x):
        return numpy.exp(x)

class VectorFunction():
    """
    Function from R^n -> R^m.
    
    In this context we use these to add additional simple terms to the objective function.
    """
    def evaluate(self, x):
        raise NotImplementedError()
    
    def jacobian(self, x):
        """
        Defaults to a finite-difference implementation.
        
        J_ij = derivative of output i with respect to input j.
        """
        output_size = self.evaluate(x).size
        result = numpy.zeros((output_size, x.size))
        for j in range(x.size):
            p = self.evaluate(x + _epsilon * 0.5)
            n = self.evaluate(x - _epsilon * 0.5)
            result[:, j] = (p - n) / _epsilon
        return result
        
class SumRegularizer(VectorFunction):
    """ Sums the input vector multiplied with a given scalar or weight vector. """
    def __init__(self, weights = 1.0):
        self.weights = weights

    def evaluate(self, x):
        return numpy.sum(self.weights * x, keepdims = True)
        
    def jacobian(self, x):
        return self.weights * numpy.ones((1, x.size))
        
class SelectRegularizer(VectorFunction):
    """ Selects a single element. """
    def __init__(self, index):
        self.index = index
        
    def evaluate(self, x):
        return x[self.index]
        
    def jacobian(self, x):
        result = numpy.zeros((1, x.size))
        result[self.index] = 1.0
        return result