import numpy

class Function():
    @classmethod
    def evaluate(x):
        raise NotImplementedError()
    
    @classmethod
    def inverse(x):
        raise NotImplementedError()
    
    @classmethod
    def derivative(x):
        raise NotImplementedError()

class HarmonicLinearRectifier(Function):
    @classmethod
    def evaluate(x):
        if x >= 0.0: return x + 1.0
        else: return 1.0 / (1.0 - x)
    
    @classmethod
    def inverse(x):
        if x >= 1.0: return x - 1.0
        else: return 1.0 - 1.0 / x
    
    @classmethod
    def derivative(x):
        if x >= 0.0: return 1.0
        else: return 1.0 / (1.0 - x) / (1.0 - x)