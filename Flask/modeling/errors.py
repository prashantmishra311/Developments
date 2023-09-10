
# this file contains the custom exceptions for modeling

class Errors(Exception):
    pass

class NonBinaryError(Errors):
    pass

class FileTypeError(Errors):
    pass

class DataFrameError(Errors):
    pass

class ColumnAbsentError(Errors):
    pass