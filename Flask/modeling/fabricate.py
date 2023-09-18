from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array, check_X_y
from sklearn import datasets
import numpy as np 
import random
import math
from typing import Callable, List, Dict, Set, Iterable, Optional, Union

from errors import *


class DiluteSparsity(BaseEstimator, TransformerMixin):
    """Class to dilute the sparsity of a data set
    """    
    def __init__(self, 
                sparsity_rate:Union[float, Dict[Union[int, str],float], List[float]], 
                dilution_rate:Optional[float]=None):
        """Constructor method for the `DiluteSparsity` class

        Args:
            sparsity_rate (Union[float, Dict[Union[int, str],float], List[float]]): Rate of sparsity for X
            dilution_rate (Optional[float], optional): Rate of dilution for y. Defaults to None.
        """        
        self.sparsity_rate = sparsity_rate
        self.dilution_rate = dilution_rate

    def fit(self, X, y=None):
        """_summary_

        Args:
            X (array like): (n_sample, n_columns)
            y (vector, optional): (n_sample,) Binary vector. Defaults to None.

        Raises:
            NonBinaryError: when y is not binary 

        Returns:
            object

        Attribute:
            X_spr (array like):  (n_sample, n_columns) converted sparse array
            y_dil (vector, NoneType): (n_sample,) Binary vector
        """        
        if y is not None:
            X_, y_ = check_X_y(X, y, dtype='numeric', order=None, 
                                copy=False, force_all_finite='allow-nan', 
                                ensure_2d=True, allow_nd=False, multi_output=False, 
                                ensure_min_samples=1, ensure_min_features=1, 
                                y_numeric=True)
            
            if not np.isin(y_,[0,1]).all():
                raise NonBinaryError("expected binary values 0/1 but y contains more than two unique values")
            else:
                y_[np.nonzero(y_)] = np.random.choice([0,1], p=[self.dilution_rate, 1-self.dilution_rate], 
                                                    size=y_.sum(), replace=True)
                self.y_dil = y_

        else:
            X_ = check_array(X, dtype='numeric', order=None, 
                            copy=False, force_all_finite='allow-nan', 
                            ensure_2d=True, allow_nd=False, 
                            ensure_min_samples=1, ensure_min_features=1)
        
        replace = (np.random.random(size=X_.shape) > self.sparsity_rate) #.astype(np.int8)
        replace[replace == False] = np.inf
        replace[replace == True] = np.nan

        self.X_spr = np.minimum(X_, replace)

        return self


if __name__ == "__main__":

    X, y = datasets.load_breast_cancer(return_X_y=True)
    print(y.sum())
    downsp = DiluteSparsity(sparsity_rate=0.2, dilution_rate=0.2)
    print(downsp.fit(X=X, y=y))
    print(downsp.y_dil.sum())

    # print(help(downsp))