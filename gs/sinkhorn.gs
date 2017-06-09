/**
 * Sinkhorn-Knopp algorithm with optional weights.
 * Computes row and column scalings r, c of an initial matrix M such that the weighted column and row sums of diag(r) * M * diag(c) are all unity, and r and c have the same harmonic means.
 *
 * @param {Range} initialMatrix The initial matrix.
 * @param {Range} rowWeights [uniform weighting] Weighting to apply to the rows (i.e. when taking a column sum). Weights are normalized to sum to 1 before the algorithm begins.
 * @param {Range} colWeights [uniform weighting] Weighting to apply to the columns (i.e. when taking a row sum). Weights are normalized to sum to 1 before the algorithm begins.
 * @param {int} numIterations [16] Number of iterations to run.
 * @return Two columns of scalings r, c such that the weighted column and row sums of diag(r) * M * diag(c) are all unity, and r and c have the same harmonic means.
 * @customfunction
 */
function sinkhorn(initialMatrix, rowWeights, colWeights, numIterations) {
    // Input checking.
    if (!isMatrix_(initialMatrix)) {
        throw "initialMatrix must be a range."
    }
    
    var initialMatrixTranspose = transposeMatrix_(initialMatrix);
    var numRows = initialMatrix.length;
    var numCols = initialMatrix[0].length;
    
    if (!rowWeights) {
        rowWeights = makeFilledArray_(numRows, 1.0 / numRows);
    } else {
        if (!isMatrix_(rowWeights)) {
            throw "rowWeights (optional) must be a range."
        }
        rowWeights = flattenMatrix_(rowWeights);
        if (rowWeights.length != numRows) {
            throw "rowWeights must have the same number of elements as initialMatrix has rows."
        }
        rowWeights = scaleArray_(rowWeights, 1.0 / sumArray_(rowWeights));
    }
    
    if (!colWeights) {
        colWeights = makeFilledArray_(numCols, 1.0 / numCols);
    } else {
        if (!isMatrix_(colWeights)) {
            throw "colWeights (optional) must be a range."
        }
        colWeights = flattenMatrix_(colWeights);
        if (colWeights.length != numCols) {
            throw "colWeights must have the same number of elements as initialMatrix has columns."
        }
        colWeights = scaleArray_(colWeights, 1.0 / sumArray_(colWeights));
    }
    
    // TODO: check zero pattern.
    
    var rowScales = makeFilledArray_(numRows, 1.0);
    var colScales = makeFilledArray_(numCols, 1.0);
    
    // TODO: number of iterations.
    if (!numIterations) {
        numIterations = 16;
    }
    
    for (iteration = 0; iteration < numIterations; iteration++) {
        // Normalize rows.
        for (var i = 0; i < numRows; i++) {
            var row = initialMatrix[i];
            var rowSum = tripleDotArrays_(row, colWeights, colScales);
            rowScales[i] = 1.0 / rowSum;
        }
        
        // Normalize columns.
        for (var j = 0; j < numCols; j++) {
            var col = initialMatrixTranspose[j];
            var colSum = tripleDotArrays_(col, rowWeights, rowScales);
            colScales[j] = 1.0 / colSum;
        }
        
        // Set relative scale.
        var rowScaleMean = weightedProduct_(rowScales, rowWeights);
        var colScaleMean = weightedProduct_(colScales, colWeights);
        var relativeScale = Math.pow(colScaleMean / rowScaleMean, 0.5);
        
        rowScales = scaleArray_(rowScales, relativeScale);
        colScales = scaleArray_(colScales, 1.0 / relativeScale);
    }
    
    return transposeMatrix_([rowScales, colScales]);
}

function isMatrix_(m) {
    return Array.isArray(m) && Array.isArray(m[0]);
}

function flattenMatrix_(m) {
    return [].concat.apply([], m);
}

function transposeMatrix_(m) {
    var numCols = 0;
    for (var i = 0; i < m.length; i++) {
        numCols = Math.max(numCols, m[i].length)
    }
    
    var result = [];
    for (var j = 0; j < numCols; j++) {
        var row = [];
        for (var i = 0; i < m.length; i++) {
            row[i] = m[i][j];
        }
        result[j] = row;
    }
    return result;
}

function makeFilledArray_(n, value) {
    var result = [];
    for (var i = 0; i < n; i++) {
        result[i] = value;
    }
    return result;
}

function sumArray_(v) {
    return v.reduce(function (prev, curr) {return prev + curr; }, 0.0);
}

function scaleArray_(v, c) {
    return v.map(function (a) {return a * c;});
}

function weightedProduct_(values, weights) {
    var result = 1.0;
    for (var i = 0; i < values.length; i++) {
        result *= Math.pow(values[i], weights[i]);
    }
    return result;
}

function dotArrays_(v0, v1) {
    var result = 0.0;
    for (var i = 0; i < v0.length; i++) {
        result += v0[i] * v1[i];
    }
    return result;
}

function tripleDotArrays_(v0, v1, v2) {
    var result = 0.0;
    for (var i = 0; i < v0.length; i++) {
        result += v0[i] * v1[i] * v2[i];
    }
    return result;
}
