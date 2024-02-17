import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, as_float_array, assert_all_finite #,_get_feature_names

from typing import Callable, List, Dict, Set, Iterable, Optional, Union

class AdaptiveThresholding(BaseEstimator, TransformerMixin):
    """class for adaptive thresholding of a probability data

    Inherits:
        sklearn.base.BaseEstimator
        sklearn.bsae.TransformerMixin
    """        
    def __init__(self, *, cutoffs:Union[float, List[float],Dict[str,float]], multiclass:bool=True, force_adaption:bool=False):
        """Constructor method for `AdaptiveThresholding` class

        Args:
            cutoffs (Union[float, List[float],Dict[str,float]]): flot/array/dict of cutoffs
            multiclass (bool, optional): whether to threshold for multiclass or multilable. 
                Defaults to True.
            force_adaption (bool, optional): forces to output in case cutoffs is incompatible 
                with prob data. Defaults to False.

        Attributes:
            column_names_in_ (Union[List[str],NoneType]): list of columns if fitted on dataframe
            y_pred_ (ndarray): transformed array after subjected to thresholding
        """        
        self._check_cutoff(cutoffs)
        self.cutoffs = cutoffs
        self.multiclass = multiclass
        self.force_adaption = force_adaption

    def fit(self, X, y=None):
        """_summary_

        Args:
            X (array): (n_sample,n_columns) array like
            y (array, optional): if a target value to incorporate in final y_pred_
                (n_sample,n_columns) for multilable 
                (n_sample,) for multiclass. 
                Defaults to None.

        Raises:
            KeyError, ValueError, Exception

        Returns:
            object
        """        

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
        
        if ((X_ >= 0)&(X_ <= 1)).all():
            if self.multiclass:
                if not np.allclose(X_.sum(axis=1), np.ones(X_.shape[0]), rtol=0.001):
                    raise ValueError("sum across columns don't sum upto 1")
                if y is not None and not np.isin(y, np.arange(X.shape[1])).all():
                    raise ValueError(f"y should be int and within {np.arange(X.shape[1])}")
            else:
                if y is not None: 
                    if y.shape != X_.shape:
                        raise Exception("X and y should have same shape in multilabel setting")
                    if not ((y >= 0)&(y <= 1)).all():
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
            self.y_pred_ = np.stack([y, X_], axis=-1).max(axis=-1) if y is not None else X_

        return self

    def _check_cutoff(self, cutoffs):
        """private method to qc cutoffs

        Args:
            cutoffs : flot/array/dict of cutoffs

        Raises:
            TypeError, ValueError
        """        
        if not isinstance(cutoffs, (float, list, np.ndarray, dict)):
            raise TypeError("cutoffs can only be float, array(float) or dict(str,float)")
        
        if isinstance(cutoffs, dict):
            values = np.asarray(list(cutoffs.values()))
        elif not isinstance(cutoffs, float):
            values = np.asarray(list(cutoffs))
        assert_all_finite(values, allow_nan=False)
        
        if not ((values >= 0)&(values <= 1)).all():
            raise ValueError("cutoffs can only be float between 0 and 1")


# test case for thresholding
if __name__ == "__main__":

    multclass = True
    if multclass:
        X = np.random.dirichlet(np.ones(5), size=1000)
        y = np.random.randint(low=0, high=5, size=X.shape[0])
    else:
        X = np.random.random(size=(1000, 5))
        y = np.random.randint(low=0, high=2, size=X.shape)
    
    cutoffs = np.random.uniform(low=0,high=1, size=5)
    
    adap_thres = AdaptiveThresholding(cutoffs=cutoffs, multiclass=multclass, force_adaption=False)
    print(adap_thres.fit(X=X, y=y))
    print(adap_thres.y_pred_)

        

