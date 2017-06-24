import csv

def write_csv(filename, data, row_headers, col_headers = None, numeric_format = '%f', *args, **kwargs):
    def iter_rows():
        yield [''] + col_headers
        for row_index in range(data.shape[0]):
            yield [row_headers[row_index]] + [(numeric_format % x) for x in data[row_index, :]]
    if col_headers is None: col_headers = row_headers
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile, *args, **kwargs)
        writer.writerows(iter_rows())