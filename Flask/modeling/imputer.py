import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, as_float_array, assert_all_finite #,_get_feature_names

from typing import Callable, List, Dict, Set, Iterable, Optional, Union

class AdaptiveThresholding(BaseEstimator, TransformerMixin):

    def __init__(self, *, cutoffs:Union[float, List[float],Dict[str,float]], multiclass:bool=True, force_adaption:bool=False):
        
        self._check_cutoff(cutoffs)
        self.cutoffs = cutoffs
        self.multiclass = multiclass
        self.force_adaption = force_adaption

    def fit(self, X, y=None):

        # self.column_names_in_ = _get_feature_names(X)
        self.column_names_in_ = getattr(X, "columns", None)
        if self.column_names_in_ is not None and isinstance(self.cutoffs, dict):
            if self.force_adaption:
                base_ = [self.cutoffs[key] if key in self.cutoffs else 0 for key in self.column_names_in_]
            else:
                try:
                    base_ = [self.cutoffs[key] for key in self.column_names_in_]
                except:
                    raise KeyError("can't find all the keys in cutoff dict")
        else:
            if self.force_adaption:
                if len(self.cutoffs) <= X.shape[1]:
                    base_ = np.pad(self.cutoffs, (0, (X.shape[1]-len(self.cutoffs))))
                else:
                    base_ = self.cutoffs[:X.shape[1]]
            else:
                if not len(self.cutoffs) == X.shape[1]:
                    raise ValueError(f"cutoffs of len {len(self.cutoffs)} can't be broadcasted to shape (,{X.shape[1]})")
                else:
                    base_ = self.cutoffs    
        
        # X should satisfy
        # numeric, finite, 2D (n_sample,n_features), min_features as 3
        X_ = check_array(X, dtype='numeric', order=None, 
                        copy=False, force_all_finite=True, 
                        ensure_2d=True, allow_nd=False, 
                        ensure_min_samples=1, ensure_min_features=3)
        
        if np.isin(np.asarray(X_),[0,1]).all():
            if self.multiclass:
                if not (X_.sum(axis=1)==1).all():
                    raise ValueError("sum across columns don't sum upto 1")
                if y is not None and not np.isin(y, np.arange(X.shape[1])):
                    raise ValueError(f"y should be int and within {np.arange(X.shape[1])}")
            else:
                if y is not None: 
                    if y.shape != X_.shape:
                        raise Exception("X and y should have same shape in multilabel setting")
                    if not np.isin(y,[0,1]).all():
                        raise ValueError("y can only be float between 0 and 1 in multilabel setting")
        else:
            raise ValueError("probability values can only be between 0 and 1")
        
        base_ = np.tile(base_, (X.shape[0],1))
        # compared = np.stack(X_, base_tiled, axis=-1).max(axis=-1)
        X_[base_ > X_] = 0

        if self.multiclass:
            y_ = X_.argmax(axis=1)
            self.y_pred_ = np.where(y==0, y_, y) if y is not None else y_
        else:
            X_[X_ > 0] = 1
            self.y_pred_ = np.stack(y, X_, axis=-1).max(axis=-1) if y is not None else X_

        return self

    def _check_cutoff(self, cutoffs):
        
        if not isinstance(cutoffs, (float, list, np.ndarray, dict)):
            raise TypeError("cutoffs can only be float, array(float) or dict(str,float)")
        
        if isinstance(cutoffs, dict):
            values = np.asarray(list(cutoffs.values()))
        elif not isinstance(cutoffs, float):
            values = np.asarray(list(cutoffs))
        assert_all_finite(values, allow_nan=False)
        print(values, np.isin(values,[0,1]))
        if not np.isin(values,[0.0,1.0]).all():
            raise ValueError("cutoffs can only be float between 0 and 1")


if __name__ == "__main__":

    X = np.random.random(size=(1000, 5))
    y_mc = np.random.randint(low=0, high=5, size=X.shape[0])
    y_ml = np.random.randint(low=0, high=2, size=X.shape)
    
    cutoffs = np.random.uniform(low=0,high=1, size=5)
    
    adap_thres = AdaptiveThresholding(cutoffs=cutoffs, multiclass=True, force_adaption=False)
    adap_thres.fit(X=X, y=None)

        

