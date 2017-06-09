import numpy
import scipy.optimize

def _processWeights(arg):
    try:
        count = arg.size
        weights = arg
    except:
        count = arg
        weights = numpy.ones((arg)) / count
    
    # replace zeros with ones for purposes of weighting the objective vector
    objectiveWeights = weights
    objectiveWeights[objectiveWeights == 0.0] = 1.0
    
    return count, weights, objectiveWeights
    
def jacobian_fd(func, shape, epsilon = None):
    if epsilon is None: epsilon = numpy.sqrt(numpy.finfo(float).eps)
    outputCount, inputCount = shape
    def result(x):
        J = numpy.zeros(shape)
        
        for inputIndex in range(inputCount):
            xdp = x.copy()
            xdp[inputIndex] += epsilon * 0.5
            
            xdn = x.copy()
            xdn[inputIndex] -= epsilon * 0.5
            
            J[:, inputIndex] = (func(xdp) - func(xdn)) / epsilon
        
        return J
    return result
    
class Balance():
    def check_jacobian(self, x = None, epsilon = None):
        if x is None: x = numpy.zeros(self.totalCount)
        J = self.jacobian(x)
        jac_fd = jacobian_fd(self.objective, J.shape, epsilon = epsilon)
        result = J - jac_fd(x)
        print('Maximum difference between evaluated Jacobian and finite difference:', numpy.max(numpy.abs(result)))
        return result

class NonSymmetricBalance(Balance):
    def __init__(self, handicapFunction, rowWeights, colWeights = None, rowDerivative = None, colDerivative = None):
        self.handicapFunction = handicapFunction
        
        if (rowDerivative is None) != (colDerivative is None):
            raise ValueError('Both rowDerivative and colDerivative must be provided for Jacobian to function.')
        
        self.rowDerivative = rowDerivative
        self.colDerivative = colDerivative
    
        self.rowCount, self.rowWeights, self.rowObjectiveWeights = _processWeights(rowWeights)
        self.colCount, self.colWeights, self.colObjectiveWeights = _processWeights(colWeights)
        
        self.totalCount = self.rowCount + self.colCount
        
    def evaluateF(self, x):
        rowHandicaps = x[:self.rowCount]
        colHandicaps = x[-self.colCount:]
        
        F = numpy.zeros((self.rowCount, self.colCount))
        
        for rowIndex in range(self.rowCount):
            for colIndex in range(self.colCount):
                F[rowIndex, colIndex] = self.handicapFunction(rowIndex, colIndex, rowHandicaps[rowIndex], colHandicaps[colIndex])
                
        return F
        
    def objective(self, x):
        F = self.evaluateF(x)
        
        # dot products are weighted 
        rowObjectives = numpy.tensordot(F, self.colWeights, axes = ([1], [0])) * self.rowObjectiveWeights
        colObjectives = numpy.tensordot(F, self.rowWeights, axes = ([0], [0])) * self.colObjectiveWeights
        
        return numpy.concatenate((rowObjectives, colObjectives))
        
    def jacobian(self, x):
        # J_ij = derivative of payoff i with respect to handicap j
        rowHandicaps = x[:self.rowCount]
        colHandicaps = x[-self.colCount:]
        
        dFdr = numpy.zeros((self.rowCount, self.colCount))
        dFdc = numpy.zeros((self.rowCount, self.colCount))
        
        for rowIndex in range(self.rowCount):
            for colIndex in range(self.colCount):
                dFdr[rowIndex, colIndex] = self.rowDerivative(rowIndex, colIndex, rowHandicaps[rowIndex], colHandicaps[colIndex])
                dFdc[rowIndex, colIndex] = self.colDerivative(rowIndex, colIndex, rowHandicaps[rowIndex], colHandicaps[colIndex])
        
        # derivative of row payoffs with respect to row handicaps
        Jrr = numpy.tensordot(dFdr, self.colWeights, axes = ([1], [0])) * self.rowObjectiveWeights
        Jrr = numpy.diag(Jrr)
        
        # derivative of col payoffs with respect to col handicaps
        Jcc = numpy.tensordot(dFdc, self.rowWeights, axes = ([0], [0])) * self.colObjectiveWeights
        Jcc = numpy.diag(Jcc)
        
        # derivative of row payoffs with respect to col handicaps
        Jrc = dFdc * self.colWeights[None, :] * self.rowObjectiveWeights[:, None]
        
        # derivative of col payoffs with respect to row handicaps
        Jcr = dFdr * self.rowWeights[:, None] * self.colObjectiveWeights[None, :]
        Jcr = numpy.transpose(Jcr)
        
        # assemble full Jacobian
        J = numpy.bmat([[Jrr, Jrc],
                        [Jcr, Jcc]])
        
        return J
        
    def optimize(self, check_jacobian_epsilon = None, *args, **kwargs):
        if self.rowDerivative is None or self.colDerivative is None:
            jac = None
        else:
            jac = self.jacobian
            
        if check_jacobian_epsilon is None:
            fun = self.objective
        else:
            epsilon = check_jacobian_epsilon
            if epsilon is True: epsilon = None
            def fun(x):
                self.check_jacobian(x, epsilon = epsilon)
                return self.objective(x)
        
        x0 = numpy.zeros((self.totalCount))
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, *args, **kwargs)
        result.rowHandicaps = result.x[:self.rowCount]
        result.colHandicaps = result.x[-self.colCount:]
        result.F = self.evaluateF(result.x)
        
        return result

class SymmetricBalance(Balance):
    def __init__(self, handicapFunction, strategyWeights, rowDerivative = None):
        self.totalCount, self.strategyWeights, self.strategyObjectiveWeights = _processWeights(strategyWeights)
        
    def evaluateF(self, x):
        F = numpy.zeros((self.totalCount, self.totalCount))
        
        for rowIndex in range(self.totalCount-1):
            for colIndex in range(rowIndex+1, self.totalCount):
                payoff = self.handicapFunction(rowIndex, colIndex, x[rowIndex], x[colIndex])
                F[rowIndex, colIndex] = payoff
                F[colIndex, rowIndex] = -payoff
                
        return F
        
    def objective(self, x):
        F = self.evaluateF(x)
        
        # dot products are weighted 
        objectives = numpy.tensordot(F, self.strategyWeights, axes = ([1], [0])) * self.strategyObjectiveWeights
        
        return objectives
        
    def jacobian(self, x):
        dFdr = numpy.zeros((self.totalCount, self.totalCount))
        
        for rowIndex in range(self.totalCount):
            for colIndex in range(self.totalCount):
                payoffDerivative = self.rowDerivative(rowIndex, colIndex, x[rowIndex], x[colIndex])
                dFdr[rowIndex, colIndex] = payoffDerivative
        
        # derivative of row payoffs with respect to row handicaps
        Jrr = numpy.tensordot(dFdr, self.strategyWeights, axes = ([1], [0])) * self.strategyObjectiveWeights
        Jrr = numpy.diag(Jrr)
        
        # derivative of row payoffs with respect to col handicaps
        dFdc = -numpy.transpose(dFdr)
        Jrc = dFdc * self.strategyWeights[None, :] * self.strategyObjectiveWeights[:, None]
        
        J = Jrr + Jrc
        
        return J
        
    def optimize(self, check_jacobian_epsilon = None, *args, **kwargs):
        if self.rowDerivative is None:
            jac = None
        else:
            jac = self.jacobian
            
        if check_jacobian_epsilon is None:
            fun = self.objective
        else:
            epsilon = check_jacobian_epsilon
            if epsilon is True: epsilon = None
            def fun(x):
                self.check_jacobian(x, epsilon = epsilon)
                return self.objective(x)
            
        x0 = numpy.zeros((self.totalCount))
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, *args, **kwargs)
        result.handicaps = result.x
        result.F = self.evaluateF(result.x)
        return result
    
class MultiplicativeBalance(NonSymmetricBalance):
    def __init__(self, initialPayoffMatrix, rowWeights = None, colWeights = None):
        self.initialPayoffMatrix = initialPayoffMatrix
        if rowWeights is None: rowWeights = initialPayoffMatrix.shape[0]
        if colWeights is None: colWeights = initialPayoffMatrix.shape[1]
    
        NonSymmetricBalance.__init__(self, self.handicapFunction, rowWeights = rowWeights, colWeights = colWeights, rowDerivative = self.rowDerivative, colDerivative = self.colDerivative)

    def handicapFunction(self, rowIndex, colIndex, rowHandicap, colHandicap):
        return self.initialPayoffMatrix[rowIndex, colIndex] * numpy.exp(colHandicap - rowHandicap) - 1.0
        
    def rowDerivative(self, rowIndex, colIndex, rowHandicap, colHandicap):
        return -self.initialPayoffMatrix[rowIndex, colIndex] * numpy.exp(colHandicap - rowHandicap)
        
    def colDerivative(self, rowIndex, colIndex, rowHandicap, colHandicap):
        return self.initialPayoffMatrix[rowIndex, colIndex] * numpy.exp(colHandicap - rowHandicap)
    
    def optimize(self, *args, **kwargs):
        result = NonSymmetricBalance.optimize(self, *args, **kwargs)
        result.rowLogHandicaps = result.rowHandicaps
        result.colLogHandicaps = result.colHandicaps
        result.rowHandicaps = numpy.exp(result.rowHandicaps)
        result.colHandicaps = numpy.exp(result.colHandicaps)
        return result
    
class LogisticSymmetricBalance(SymmetricBalance):
    def __init__(self, initialPayoffMatrix, strategyWeights = None):
        self.offsetMatrix = numpy.log(1.0 / initialPayoffMatrix - 1.0)
        if strategyWeights is None: strategyWeights = initialPayoffMatrix.shape[0]
        SymmetricBalance.__init__(self, self.handicapFunction, strategyWeights, rowDerivative = self.rowDerivative)
    
    def handicapFunction(self, rowIndex, colIndex, rowHandicap, colHandicap):
        offset = self.offsetMatrix[rowIndex, colIndex]
        return 1.0 / (1.0 + numpy.exp(rowHandicap - colHandicap + offset)) - 0.5
        
    def rowDerivative(self, rowIndex, colIndex, rowHandicap, colHandicap):
        payoff = self.handicapFunction(rowIndex, colIndex, rowHandicap, colHandicap)
        return payoff * payoff - 0.25