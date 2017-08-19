import numpy

class Dataset():
    def __init__(self, data, row_names, col_names = None, symmetric = False):
        self.data = data
        self.row_names = row_names
        if col_names is None: col_names = row_names
        self.col_names = col_names
        self.symmetric = symmetric
        
    def sorted_by_sum(self):
        data = self.data
        idx_row = numpy.argsort(self.data.sum(axis = 1))[::-1]
        data = data[idx_row, :]
        row_names = list(self.row_names[i] for i in idx_row)
        
        # Reverse order, so best to worst.
        if self.symmetric:
            idx_col = idx_row
        else:
            idx_col = numpy.argsort(self.data.sum(axis = 0))
        data = data[:, idx_col]
        col_names = list(self.col_names[i] for i in idx_col)
        
        return Dataset(data, row_names, col_names)