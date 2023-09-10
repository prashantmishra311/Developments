import pandas as pd 
import numpy as np 
from typing import List, Optional, Set, Callable, Union 
import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets

from errors import *


class Fabrication(object):

    def __init__(self, data:pd.DataFrame, features:List[str], target:str):

        if not isinstance(data, pd.DataFrame):
            raise DataFrameError("Invalid data type, only dataframe allowed")
        else:
            self.data = data

        self.features = features
        self.target = target

        if not set(self.features + [self.target]) <= set(self.data.columns.tolist()):
            absent = set(self.features + [self.target]) - set(self.data.columns.tolist())
            raise ColumnAbsentError("Columns not found: ", ", ".join(list(absent)))
    
    def make_sparse(self, dilution_rate:float = 0.5, sparsity_rate:Union[float, Dict]=0.2, exclusion:Optional[list]=None):

        pass


# def fabricate_data(data:pd.DataFrame, features:List[str], target:str):



