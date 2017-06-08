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

def nonSymmetric(handicapFunction, rowWeights, colWeights = None, rowDerivative = None, colDerivative = None, *args, **kwargs):
    def evaluateF(x):
        rowHandicaps = x[:rowCount]
        colHandicaps = x[-colCount:]
        
        F = numpy.zeros((rowCount, colCount))
        
        for rowIndex in range(rowCount):
            for colIndex in range(colCount):
                F[rowIndex, colIndex] = handicapFunction(rowIndex, colIndex, rowHandicaps[rowIndex], colHandicaps[colIndex])
                
        return F
        
    def objective(x):
        F = evaluateF(x)
        
        # dot products are weighted 
        rowObjectives = numpy.tensordot(F, colWeights, axes = ([1], [0])) * rowObjectiveWeights
        colObjectives = numpy.tensordot(F, rowWeights, axes = ([0], [0])) * colObjectiveWeights
        
        return numpy.concatenate((rowObjectives, colObjectives))
    
    rowCount, rowWeights, rowObjectiveWeights = _processWeights(rowWeights)
    colCount, colWeights, colObjectiveWeights = _processWeights(colWeights)
    
    if (rowDerivative is None) != (colDerivative is None):
        raise ValueError('Both rowDerivative and colDerivative must be provided for Jacobian to function.')
    elif rowDerivative is None:
        jac = None
    else:
        jac = None # TODO
        
    x0 = numpy.zeros((rowCount + colCount))
    result = scipy.optimize.root(fun = objective, x0 = x0, jac = jac, *args, **kwargs)
    result.rowHandicaps = result.x[:rowCount]
    result.colHandicaps = result.x[-colCount:]
    result.F = evaluateF(result.x)
    
    return result

def symmetric(handicapFunction, strategyWeights, strategyDerivative = None, *args, **kwargs):
    def evaluateF(x):
        F = numpy.zeros((strategyCount, strategyCount))
        
        for rowIndex in range(strategyCount-1):
            for colIndex in range(rowIndex+1, strategyCount):
                payoff = handicapFunction(rowIndex, colIndex, x[rowIndex], x[colIndex])
                F[rowIndex, colIndex] = payoff
                F[colIndex, rowIndex] = -payoff
                
        return F
        
    def objective(x):
        F = evaluateF(x)
        
        # dot products are weighted 
        objectives = numpy.tensordot(F, strategyWeights, axes = ([1], [0])) * strategyObjectiveWeights
        
        return objectives
        
    strategyCount, strategyWeights, strategyObjectiveWeights = _processWeights(strategyWeights)
        
    if strategyDerivative is None:
        jac = None
    else:
        jac = None # TODO
    
    x0 = numpy.zeros((strategyCount))
    result = scipy.optimize.root(fun = objective, x0 = x0, jac = jac, *args, **kwargs)
    result.handicaps = result.x
    result.F = evaluateF(result.x)
    
    return result

def multiplicative(initialPayoffMatrix, rowWeights = None, colWeights = None, *args, **kwargs):
    def handicapFunction(rowIndex, colIndex, rowHandicap, colHandicap):
        return initialPayoffMatrix[rowIndex, colIndex] * numpy.exp(colHandicap - rowHandicap) - 1.0
    
    if rowWeights is None: rowWeights = initialPayoffMatrix.shape[0]
    if colWeights is None: colWeights = initialPayoffMatrix.shape[1]
    
    return nonSymmetric(handicapFunction, rowWeights, colWeights, *args, **kwargs)
    
def logisticSymmetric(initialPayoffMatrix, strategyWeights = None, *args, **kwargs): 
    def handicapFunction(rowIndex, colIndex, rowHandicap, colHandicap):
        offset = offsetMatrix[rowIndex, colIndex]
        return 1.0 / (1.0 + numpy.exp(rowHandicap - colHandicap + offset)) - 0.5
    
    offsetMatrix = numpy.log(1.0 / initialPayoffMatrix - 1.0)
    
    # TODO: derivatives
    
    if strategyWeights is None: strategyWeights = initialPayoffMatrix.shape[0]

    return symmetric(handicapFunction, strategyWeights, *args, **kwargs) 