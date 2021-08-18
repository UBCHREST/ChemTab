# main.py imports
from data.pre_processing import DataPreparer
from data.train_test_manager import DataManager
from experiment_executor.gp_experiment_executor import GPExperimentExecutor
from experiment_executor.simple_dnn_experiment_executor import DNNExperimentExecutor
from experiment_executor.pcdnn_v1_experiment_executor import PCDNNV1ExperimentExecutor
from experiment_executor.pcdnn_v2_experiment_executor import PCDNNV2ExperimentExecutor
from models.gpmodel import GPModel
from models.simplednn import SimpleDNNModel
from models.pcdnnv1 import PCDNNV1Model
from models.pcdnnv2 import PCDNNV2Model
import pandas as pd


# baseline imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.inspection import plot_partial_dependence
from sklearn.base import BaseEstimator, RegressorMixin
import scipy.stats as stats
import pandas as pd
import numpy as np
import os

from models.gpmodel import CustomGPR

def do_perm_feature_importance(model, X_data=None, Y_data=None,
                               data_manager=None, n_repeats=30, random_state=0):
    assert not (X_data is None and Y_data is None and data_manager is None)
    if data_manager is not None:
        X_data = pd.DataFrame(data_manager.X_test, columns=data_manager.input_data_cols)
        Y_data = pd.DataFrame(data_manager.Y_test, columns=data_manager.output_data_cols)
    #pdb.set_trace()
    
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_data, Y_data,
                               n_repeats=n_repeats,
                               random_state=random_state,
                               scoring='neg_mean_squared_error')

    argsort = r.importances_mean.argsort()[::-1]
    
    for i in argsort:
         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{list(X_data.columns)[i]}; {r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}")
    a = list(X_data.columns[argsort])
    b = r.importances_mean[argsort]
    c = r.importances_std[argsort]
    bar = plt.bar(a, b, label='mean')
    err = plt.errorbar(a, b, yerr=c, fmt="o", color="r", label='std')
    plt.yscale('log')
    plt.title('Model\'s (Permutation) Feature Importance')
    plt.xticks(rotation = 90)
    plt.legend(handles=[bar, err])
    plt.show()


class NNWrapper(BaseEstimator, RegressorMixin):
    """Wraps Amol's NN classes to comform to the interface expected by SciPy"""
    def __init__(self, model):
        super().__init__()
        self._model = model
        self._input_names = [i.name for i in model.inputs]
        self._has_zmix = 'Zmix' in  self._input_names
        
        # this tells sklearn that the model is fitted apparently... (as of version 1.6.2)
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py
        self._I_am_fitted_ = 'nonsense_value'
    
    def get_XY_data(self, dm):
        """
        Extracts appropriate X & Y data from data-manager for evaluation of our model
        (assumes that datamanager was already called to create relevant dataset)
        """
        X_data = np.concatenate((dm.X_test, dm.zmix_test.reshape([-1, 1])), axis=1) if self._has_zmix else dm.X_test
        extra_input_cols = ['Zmix'] if self._has_zmix else []
        Y_data = dm.Y_test
        return pd.DataFrame(X_data.astype('f8'), columns=list(dm.input_data_cols)+extra_input_cols), pd.DataFrame(Y_data.astype('f8'), columns=dm.output_data_cols)
    
    def _get_input_dict(self, X_data):
        X_data = np.asarray(X_data)
        
        # self._input_names[0] == 'species_input' or 'inputs' depending on NN version
        input_dict = {self._input_names[0]: X_data[:,:-1]}
        if self._has_zmix: input_dict['Zmix'] = X_data[:,-1]
        return input_dict
    
    def predict(self, X_data):
        return self._model.predict(self._get_input_dict(X_data))
    
    def get_params(self, deep=True):
        return self._model.get_weights()
    
    def fit(self, X_data, Y_data):
        self._model.fit(self._get_input_dict(X_data), Y_data, batch_size=32)

# original CustomGPR class is already compatible with scipy so we just need to add get_XY_data method...
class GP_wrapper(CustomGPR):
    def __init__(self, gp_model):
        # integrate gp_model into this class (i.e. this class 'becomes' gp model)
        vars(self).update(vars(gp_model))
        
    def get_XY_data(self, dm):
        """
        Extracts appropriate X & Y data from data-manager for evaluation of our model
        (assumes that datamanager was already called to create relevant dataset)
        """
        X_data = pd.DataFrame(dm.X_test, columns=dm.input_data_cols)
        Y_data = pd.DataFrame(dm.Y_test, columns=dm.output_data_cols)
        return X_data, Y_data
