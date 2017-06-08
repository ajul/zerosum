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
        
    # TODO: derivative
    x0 = numpy.zeros((self.rowCount + self.colCount))
    result = scipy.optimize.least_squares(fun = objective, x0 = x0, *args, **kwargs)
    result.rowHandicaps = result.x[:rowCount]
    result.colHandicaps = result.x[-colCount:]
    result.F = evaluateF(result.x)
    
    return result

def symmetric(handicapFunction, strategyWeights, strategyDerivative = None, *args, **kwargs):
    def evaluateF(x):
        F = numpy.zeros((strategyCount, strategyCount))
        
        for rowIndex in range(strategyCount):
            for colIndex in range(rowIndex+1, strategyCount):
                payoff = handicapFunction(rowIndex, colIndex, handicaps[rowIndex], handicaps[colIndex])
                F[rowIndex, colIndex] = payoff
                F[colIndex, rowIndex] = -payoff
                
        return F
        
    def objective(x):
        F = evaluateF(x)
        
        # dot products are weighted 
        objectives = numpy.tensordot(F, strategyWeights, axes = ([1], [0])) * strategyObjectiveWeights
        
        return objectives
        
    strategyCount, strategyWeights, strategyObjectiveWeights = _processWeights(strategyWeights)
        
    # TODO: derivative
    
    x0 = numpy.zeros((self.rowCount))
    result = scipy.optimize.least_squares(fun = objective, x0 = x0, *args, **kwargs)
    result.handicaps = result.x
    result.F = evaluateF(result.x)
    
    return result