import pandas as pd
import numpy as np
import multiprocessing as mp
import lightgbm as lgb
import copy
import tqdm
from sklearn.model_selection import StratifiedKFold
from scipy.special import softmax
#from tsfeatures import tsfeatures
from math import isclose

from utils_input import _check_valid_df, _check_same_type, _check_passed_dfs, _check_valid_columns
from utils_models import _train_lightgbm, _train_lightgbm_cv, _train_lightgbm_cv_optimal_params


class FFORMA:

    def __init__(self, objective='FFORMA', verbose_eval=True,
                 early_stopping_rounds=10,
                 params=None,
                 param_grid=None,
                 use_cv=False, nfolds=5,
                 greedy_search=False,
                 threads=None, seed=260294, CI = False, original_article = False):

        """ Feature-based Forecast Model Averaging.

        Python Implementation of FFORMA.

        ** References: **
        <https://robjhyndman.com/publications/fforma/>
        """
        self.dict_obj = {'FFORMA': (self.fforma_objective, self.fforma_loss)}
        self.original_article = original_article
        self.CI = CI
        fobj, feval = self.dict_obj.get(objective, (None, None))
        self.objective, self.greedy_search = objective, greedy_search

        if threads is None:
            threads = mp.cpu_count() - 1

        init_params = {
            'objective': 'multiclass',
            'nthread': threads,
            'silent': 1,
            'seed': seed
        }

        if params:
            train_params = {**params, **init_params}
        else:
            train_params = {'n_estimators': 100}
            train_params = {**train_params, **init_params}


        if param_grid is not None:
            pass
        
        elif use_cv:
            folds = lambda holdout_feats, best_models: StratifiedKFold(n_splits=nfolds).split(holdout_feats, best_models)

            self._train = lambda holdout_feats, best_models, greedy_search, strat: _train_lightgbm_cv(holdout_feats, best_models,
                                                                                train_params, fobj, feval,
                                                                                early_stopping_rounds, verbose_eval,
                                                                                seed, folds)
        else:
            self._train = lambda holdout_feats, best_models, greedy_search, strat: _train_lightgbm(holdout_feats, best_models,
                                                                             train_params, fobj, feval,
                                                                             early_stopping_rounds, verbose_eval,
                                                                             seed, greedy_search, strat)
        self._fitted = False

    def _tsfeatures(self, y_train_df, y_val_df, freq):
        complete_data = pd.concat([y_train_df, y_test_df.filter(items=['unique_id', 'ds', 'y'])])
        holdout_feats = tsfeatures(y_train_df)
        feats = tsfeatures(complete_data)

        return feats, holdout_feats

    def fforma_objective(self, predt: np.ndarray, dtrain) -> (np.ndarray, np.ndarray):
        #labels of the elements in the training set
        y = dtrain.get_label().astype(int)        
        n_train = len(y)
        self.y_obj = y

        #predictions of the weights, start off with 0,0,0,0..
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')

        #softmax transform of the weights, start of with 1/7, 1/7
        preds_transformed = softmax(preds, axis=1)

        if self.CI | self.original_article:
        #Old weighted_avg_loss_func 
            weighted_avg_loss_func = (preds_transformed*self.contribution_to_error[y, :]).sum(axis=1).reshape((n_train, 1))
            grad = preds_transformed*(self.contribution_to_error[y, :] - weighted_avg_loss_func)
            hess = self.contribution_to_error[y,:]*preds_transformed*(1.0-preds_transformed) - grad*preds_transformed

        else:
            #Changed to use all individual errors. (new)
            number_of_periods = int(len(self.errors_full)/len(self.contribution_to_error))
            preds_transformed_new = np.repeat(preds_transformed, number_of_periods, axis = 0) #Yes
            weighted_avg_loss_func = np.abs((preds_transformed_new*self.errors_full.loc[y].reindex(y, level = 0)).sum(axis = 1)).groupby('unique_id').mean().values.reshape((n_train, 1)) #Yes
            weighted_avg_loss_func_ungrouped = np.abs((preds_transformed_new*self.errors_full.loc[y].reindex(y, level = 0)).sum(axis = 1)) #Yes            
            grad = preds_transformed*((np.abs(self.errors_full.loc[y].reindex(y, level = 0)) - np.array([weighted_avg_loss_func_ungrouped.values]).T).groupby('unique_id')[self.errors_full.columns].mean().values)
            hess = (np.abs(self.errors_full.loc[y].reindex(y, level=0))*preds_transformed_new*(1-preds_transformed_new)).groupby('unique_id')[self.errors_full.columns].mean().values - grad*preds_transformed

            #hess = grad*(1 - 2*preds_transformed) #True Hessian (has numerical errors apparently?)        
        return grad.flatten('F'), hess.flatten('F')

    def fforma_loss(self, predt: np.ndarray, dtrain) -> (str, float):
        y = dtrain.get_label().astype(int)
        self.y_loss = y
        n_train = len(y)
        
        #for lightgbm
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')
        
        #lightgbm uses margins!
        preds_transformed = softmax(preds, axis=1)
        number_of_periods = int(len(self.errors_full)/len(self.contribution_to_error))

        if self.CI | self.original_article:
            #Old
            weighted_avg_loss_func = (preds_transformed*self.contribution_to_error[y, :]).sum(axis=1)
            fforma_loss = weighted_avg_loss_func.mean()
        else:
            #Changed these lines to complete individual errors (new)
            preds_transformed_new = np.repeat(preds_transformed, number_of_periods, axis = 0)
            weighted_avg_loss_func = np.abs((preds_transformed_new*self.errors_full.loc[y].reindex(y, level = 0)).sum(axis = 1)).groupby('unique_id').mean().values.reshape((n_train, 1))
            fforma_loss = weighted_avg_loss_func.mean()
                
        return 'FFORMA-loss', fforma_loss, False

    def fit(self, y_train_df=None, y_val_df=None,
            val_periods=None,
            errors=None, holdout_feats=None,
            feats=None, freq=None, base_model=None,
            sorted_data=False):
        """
        y_train_df: pandas df
            panel with columns unique_id, ds, y
        y_val_df: pandas df
            panel with columns unique_id, ds, y, {model} for each model to ensemble
        val_periods: int or pandas df
            int: number of val periods
            pandas df: panel with columns unique_id, val_periods
        """

        if (errors is None) and (feats is None):
            assert (y_train_df is not None) and (y_val_df is not None), "you must provide a y_train_df and y_val_df"
            is_pandas_df = self._check_passed_dfs(y_train_df, y_val_df_)

            if not sorted_data:
                if is_pandas_df:
                    y_train_df = y_train_df.sort_values(['unique_id', 'ds'])
                    y_val_df = y_val_df.sort_values(['unique_id', 'ds'])
                else:
                    y_train_df = y_train_df.sort_index()
                    y_val_df = y_val_df.sort_index()

        if errors is None:
            pass
        else:
            if self.original_article:
                errors = np.abs(errors).groupby('unique_id')[errors.columns].mean()

            self.errors_full = errors.copy()
            
            if self.CI:
                self.contribution_to_error = np.abs(errors).groupby('unique_id')[errors.columns].mean().values
                self.best_models = self.contribution_to_error.argmin(axis = 1)
                best_models = self.contribution_to_error.argmin(axis =1)

            else:
                errors = np.abs(errors).groupby('unique_id')[errors.columns].mean()
                self.contribution_to_error = errors.values
                best_models = self.contribution_to_error.argmin(axis=1)
                self.best_models = best_models


        if feats is None:
            feats, holdout_feats = self._tsfeatures(y_train_df, y_val_df, freq)
        else:
            assert holdout_feats is not None, "when passing feats you must provide holdout feats"
            _check_valid_columns(feats, cols=['unique_id'], cols_index=['unique_id'])

        self.lgb = self._train(holdout_feats, best_models, greedy_search = False, strat = None)

        raw_score_ = self.lgb.predict(feats, raw_score=True)
        self.raw_score_ = pd.DataFrame(raw_score_,
                                       index=feats.index,
                                       columns=errors.columns)

        weights = softmax(raw_score_, axis=1)
        self.weights_ = pd.DataFrame(weights,
                                     index=feats.index,
                                     columns=errors.columns)

        #This is adding new layer, that drops models if they do not add to performance!
        if self.greedy_search:
            performance = self.lgb.best_score['valid_1']['FFORMA-loss']
            improvement = True
            errors_1 = copy.deepcopy(self.errors_full)
            print(f'\nInitial performance: {performance}\n')
            while improvement and errors.shape[1]>2:
                model_to_remove = self.weights_.mean().nsmallest(1).index
                print(f'Removing {model_to_remove}\n')
                errors_1 = errors_1.drop(columns=model_to_remove)

                #reset errors_full
                self.errors_full = errors_1

                #reset contribution_to_error
                self.contribution_to_error = np.abs(errors_1).groupby('unique_id')[errors_1.columns].mean().values

                #reset best_models
                best_models = self.contribution_to_error.argmin(axis=1)

                print(best_models)

                new_lgb = self._train(holdout_feats, best_models, greedy_search = True, strat = self.best_models)
                performance_new_lgb = new_lgb.best_score['valid_1']['FFORMA-loss']
                better_model = performance_new_lgb <= performance
                if not better_model:
                    print(f'\nImprovement not reached: {performance_new_lgb}')
                    print('stopping greedy_search')
                    improvement = False
                else:
                    performance = performance_new_lgb
                    print(f'\nReached better performance {performance}\n')
                    self.lgb = new_lgb
                    self.obj_y_fin = self.y_obj

                    raw_score_ = self.lgb.predict(feats, raw_score=True)
                    self.raw_score_ = pd.DataFrame(raw_score_,
                                                   index=feats.index,
                                                   columns=errors_1.columns)

                    weights = softmax(raw_score_, axis=1)
                    self.weights_ = pd.DataFrame(weights,
                                                 index=feats.index,
                                                 columns=errors_1.columns)
        self._fitted = True


    def predict(self, y_hat_df, fforms=False):
        """
        Parameters
        ----------
        y_hat_df: pandas df
            panel with columns unique_id, ds, {model} for each model to ensemble
        """
        assert self._fitted, "Model not fitted yet"

        if fforms:
            weights = (self.weights_.div(self.weights_.max(axis=1), axis=0) == 1)*1
            name = 'fforms_prediction'
        else:
            weights = self.weights_
            name = 'fforma_prediction'
        fforma_preds = weights * y_hat_df
        fforma_preds = fforma_preds.sum(axis=1)
        fforma_preds.name = name
        preds = pd.concat([y_hat_df, fforma_preds], axis=1)

        return preds
