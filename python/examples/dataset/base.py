class Dataset():
    def __init__(self, data, row_names, col_names = None):
        self.data = data
        self.row_names = row_names
        if col_names is None: col_names = row_names
        self.col_names = col_names
        
    def sorted_by_sum(self):
        data = self.data
        idx_row = numpy.argsort(self.data.sum(axis = 1))
        data = data[idx_row, :]
        row_names = list(self.row_names[i] for i in idx_row)
        
        # Reverse order, so best to worst.
        idx_col = numpy.argsort(self.data.sum(axis = 0))[::-1]
        data = data[:, idx_col]
        col_names = list(self.col_names[i] for i in idx_col)
        return Dataset(data, row_names, col_names)