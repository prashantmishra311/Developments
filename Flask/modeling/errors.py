import warnings
# this file contains the custom exceptions for modeling
__all__ = [
            "Errors",
            "NonBinaryError",
            "FileTypeError",
            "DataFrameError",
            "ColumnAbsentError"
        ]
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
