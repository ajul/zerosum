import csv

def write_csv(filename, data, row_headers, col_headers = None, row_footers = None, col_footers = None, numeric_format = '%f', *args, **kwargs):
    def iter_rows():
        yield [''] + col_headers
        for row_index in range(data.shape[0]):
            result = [row_headers[row_index]] + [(numeric_format % x) for x in data[row_index, :]]
            if row_footers is not None:
                result.append(numeric_format % row_footers[row_index])
            yield result
        if col_footers is not None:
            yield [''] + [numeric_format % col_footer for col_footer in col_footers]
    if col_headers is None: col_headers = row_headers
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile, *args, **kwargs)
        writer.writerows(iter_rows())