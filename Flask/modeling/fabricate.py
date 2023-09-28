from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
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

    def _fit(self, X, y=None):
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
        self.column_names_in_ = getattr(X, 'columns', None)
        
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
        
        replace = (np.random.random(size=X_.shape) > self.sparsity_rate).astype(int)

        self.X_spr = np.where(replace == 0, np.nan, X_)

        return self
    
    def fit(self, X, y=None):
        """method for fitting the data

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
        self.column_names_in_ = getattr(X, 'columns', None)
        if y is not None:
            X_, y_ = check_X_y(X, y, dtype='numeric', order=None, 
                                copy=False, force_all_finite='allow-nan', 
                                ensure_2d=True, allow_nd=False, multi_output=False, 
                                ensure_min_samples=1, ensure_min_features=1, 
                                y_numeric=True)
            
            if not np.isin(y_,[0,1]).all():
                raise NonBinaryError("expected binary values 0/1 but y contains more than two unique values")

        else:
            X_ = check_array(X, dtype='numeric', order=None, 
                            copy=False, force_all_finite='allow-nan', 
                            ensure_2d=True, allow_nd=False, 
                            ensure_min_samples=1, ensure_min_features=1)
        
        self.n_feats_in_ = X_.shape[1]

        return self
    
    def transform(self, X, y=None):
        """method to transform the data

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
        
        check_is_fitted(self)
        if hasattr(self, 'column_names_in_'):
            if self.column_names_in_ is not None:
                _column_names_tran = getattr(X, 'columns', None)
                if _column_names_tran is None:
                    raise ColumnAbsentError(f"fitted on {len(self.column_names_in_)} columns but found None in X")
                elif np.isin(self.column_names_in_,_column_names_tran).all():
                    if len(set(_column_names_tran))>len(set(self.column_names_in_)):
                        warnings.warn("additional columns found, ignoring their transformation")
                    X = X[self.column_names_in_]
                else:
                    raise ColumnAbsentError(f"fitted on {len(self.column_names_in_)}"
                                            f" columns but found only {len(_column_names_tran)} in X")
                    
            else:
                if not (X.shape[1]==self.n_feats_in_):
                    raise ValueError(f"expected shape (,{self.n_feats_in_}) but found (,{X.shape[1]})")
        
        else:
            AttributeError(f"{self} object doesn't have `column_names_in_` attribute, "
                          "check if fitted properly")
        
        return self._check_and_transform(X, y)
        
    def _check_and_transform(self, X, y=None):
        
        if y is not None:
            X_, y_ = check_X_y(X, y, dtype='numeric', order=None, 
                                copy=False, force_all_finite='allow-nan', 
                                ensure_2d=True, allow_nd=False, multi_output=False, 
                                ensure_min_samples=1, ensure_min_features=1, 
                                y_numeric=True)
            replace = (np.random.random(size=X_.shape) > self.sparsity_rate).astype(int)
            X_tran = np.where(replace == 0, np.nan, X_)
            
            if not np.isin(y_,[0,1]).all():
                raise NonBinaryError("expected binary values 0/1 but y contains more than two unique values")
            else:
                if self.dilution_rate is None:
                    setattr(self, 'dilution_rate', 0)
                y_[np.nonzero(y_)] = np.random.choice([0,1], p=[self.dilution_rate, 1-self.dilution_rate], 
                                                    size=y_.sum(), replace=True)
                y_tran = y_
            
            return X_tran, y_tran

        else:
            X_ = check_array(X, dtype='numeric', order=None, 
                            copy=False, force_all_finite='allow-nan', 
                            ensure_2d=True, allow_nd=False, 
                            ensure_min_samples=1, ensure_min_features=1)
        
            replace = (np.random.random(size=X_.shape) > self.sparsity_rate).astype(int)
            X_tran = np.where(replace == 0, np.nan, X_)

            return X_tran


if __name__ == "__main__":

    X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=False)
    downsp = DiluteSparsity(sparsity_rate=0.2, dilution_rate=0.2)
    downsp.fit(X=X, y=y)
    print(y.sum())
    X_,y_ = downsp.transform(X=X, y=y)
    print(y_.sum())
