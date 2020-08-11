from math import sqrt
import numpy as np
from numpy.random import seed
seed(42)
import pandas as pd
import gc
from statsmodels.tsa.api import ExponentialSmoothing
from tbats import TBATS
import pmdarima as pm
from itertools import product
import multiprocessing
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")



def detrend(insample_data):
    """
    Calculates a & b parameters of LRL
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b

def deseasonalize(original_ts, ppy):
    """
    Calculates and returns seasonal indices
    """
    
    original_ts = np.array(original_ts)
    
    if seasonality_test(original_ts, ppy):

        #Get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        #Get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        si = np.ones(ppy)

    return si

def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS
    """

    ts_init = pd.Series(ts_init)
    
    if len(ts_init) % 2 == 0:
        ts_ma = ts_init.rolling(window, center=True).mean()
        ts_ma = ts_ma.rolling(2, center=True).mean()
        ts_ma = np.roll(ts_ma, -1)
    else:
        ts_ma = ts_init.rolling(window, center=True).mean()

    return ts_ma


def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    """
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(data, k):
    """
    Autocorrelation function
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)


class Naive:
    """
    Naive model.
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init):
        self.ts_naive = [ts_init[-1]]
        return self

    def predict(self, h, CI):
        return np.array(self.ts_naive * h)
    
class SeasonalNaive:
    """
    Seasonal Naive model.
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.ts_seasonal_naive = ts_init[-frcy:]
        return self

    def predict(self, h, CI):
        repetitions = int(np.ceil(h/len(self.ts_seasonal_naive)))
        y_hat = np.tile(self.ts_seasonal_naive, reps=repetitions)[:h]
        return y_hat

class Naive2:
    """
    Naive2: Naive after deseasonalization.
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        seasonality_in = deseasonalize(ts_init, frcy)
        repetitions = int(np.ceil(len(ts_init) / frcy))
        
        self.ts_init = ts_init
        self.s_hat = np.tile(seasonality_in, reps=repetitions)[:len(ts_init)]
        self.ts_des = ts_init / self.s_hat
                
        return self
    
    def predict(self, h, CI):
        s_hat = SeasonalNaive().fit(self.s_hat, frcy=self.frcy).predict(h, CI)
        r_hat = Naive().fit(self.ts_des).predict(h, CI)        
        y_hat = s_hat * r_hat
        return y_hat

class RandomWalkDrift:
    """
    RandomWalkDrift: Random Walk with drift, no seasonality.
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init):
        self.drift = (ts_init[-1] - ts_init[0])/(len(ts_init)-1)
        self.naive = [ts_init[-1]]
        return self

    def predict(self, h, CI):
        naive = np.array(self.naive * h)
        drift = self.drift*np.array(range(1,h+1))
        y_hat = naive + drift
        y_hat[y_hat<0] = 0
        return y_hat
        
class ETS:
    """
    ETS: Exponential Smoothing, with seasonal pattern
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.ets = ExponentialSmoothing(
            ts_init[-36:], 
            seasonal='add',
            seasonal_periods=self.frcy
        ).fit()
        return self
    
    def predict(self, h, CI):     
        y_hat = self.ets.forecast(steps=h)        
        return y_hat
    
class ETS_NS:
    """
    ETS_NS: Exponential Smoothing, without seasonal pattern
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.ets = ExponentialSmoothing(
                ts_init[-36:], 
                trend='add', 
            ).fit()
        return self
    
    def predict(self, h, CI):     
        y_hat = self.ets.forecast(steps=h)        
        return y_hat    

class TBATSFF:
    """
    TBATS: Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components. 
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.tbats = TBATS(seasonal_periods=[self.frcy], use_arma_errors=False, show_warnings = False).fit(ts_init[-36:])
        
        return self
    
    def predict(self, h, CI):
        y_hat, forecast_int = self.tbats.forecast(steps=h, confidence_level=0.95)

        if CI:
            lower_b = forecast_int['lower_bound']
            upper_b = forecast_int['upper_bound']
            forecast_int_fin = [[x, y] for (x,y) in zip(lower_b, upper_b)]
            return forecast_int_fin
        else:
            return y_hat
    
    
class DTRENDS:
    """
    Deterministic Trend with seasonality
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.dtrends = sm.tsa.UnobservedComponents(ts_init[-48:], level = 'dtrend', seasonal = self.frcy).fit(method = 'powell', disp = False)
        return self
    
    def predict(self, h, CI):
        
        forecast_df = self.dtrends.get_forecast(h).summary_frame(alpha = 0.05)
        y_hat = forecast_df['mean'].tolist()
        lower_b = forecast_df['mean_ci_lower'].tolist()
        upper_b = forecast_df['mean_ci_upper'].tolist()
        forecast_int_fin = [[x, y] for (x,y) in zip(lower_b, upper_b)]
        
        if CI:
            return forecast_int_fin
        else:
            return y_hat
    
class LLDTRENDS:
    """
    Local linear deterministic trend with seasonality.  
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.lldtrends = sm.tsa.UnobservedComponents(ts_init[-48:], level = 'lldtrend', seasonal = self.frcy).fit(method = 'powell', disp = False)
        return self
    
    def predict(self, h, CI):
        
        forecast_df = self.lldtrends.get_forecast(h).summary_frame(alpha = 0.05)
        y_hat = forecast_df['mean'].tolist()
        lower_b = forecast_df['mean_ci_lower'].tolist()
        upper_b = forecast_df['mean_ci_upper'].tolist()
        forecast_int_fin = [[x, y] for (x,y) in zip(lower_b, upper_b)]
        
        if CI:
            return forecast_int_fin
        else:
            return y_hat    
    
class RWDriftS:
    """
    Random walk with drift and seasonality
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.rwalkds = sm.tsa.UnobservedComponents(ts_init[-48:], level = 'rwdrift', seasonal = self.frcy).fit(method = 'powell', disp = False)
        return self
    
    def predict(self, h, CI):
        
        forecast_df = self.rwalkds.get_forecast(h).summary_frame(alpha = 0.05)
        y_hat = forecast_df['mean'].tolist()
        lower_b = forecast_df['mean_ci_lower'].tolist()
        upper_b = forecast_df['mean_ci_upper'].tolist()
        forecast_int_fin = [[x, y] for (x,y) in zip(lower_b, upper_b)]
        
        if CI:
            return forecast_int_fin
        else:
            return y_hat    

    
class LLevelS:
    """
    Local Level with seasonality
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.llevels = sm.tsa.UnobservedComponents(ts_init[-48:], level = 'llevel', seasonal = self.frcy).fit(method = 'powell', disp = False)
        return self
    
    def predict(self, h, CI):
        forecast_df = self.llevels.get_forecast(h).summary_frame(alpha = 0.05)
        y_hat = forecast_df['mean'].tolist()
        lower_b = forecast_df['mean_ci_lower'].tolist()
        upper_b = forecast_df['mean_ci_upper'].tolist()
        forecast_int_fin = [[x, y] for (x,y) in zip(lower_b, upper_b)]
        
        if CI:
            return forecast_int_fin
        else:
            return y_hat    
    
class DTREND:
    """
    Deterministic trend
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.dtrends = sm.tsa.UnobservedComponents(ts_init[-48:], level = 'dtrend', seasonal = 1).fit(method = 'powell', disp = False)
        return self
    
    def predict(self, h, CI):
        forecast_df = self.dtrends.get_forecast(h).summary_frame(alpha = 0.05)
        y_hat = forecast_df['mean'].tolist()
        lower_b = forecast_df['mean_ci_lower'].tolist()
        upper_b = forecast_df['mean_ci_upper'].tolist()
        forecast_int_fin = [[x, y] for (x,y) in zip(lower_b, upper_b)]
        
        if CI:
            return forecast_int_fin
        else:
            return y_hat    
    
class RWDrift:
    """
    Random Walk with Drift
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.rwalkds = sm.tsa.UnobservedComponents(ts_init[-48:], level = 'rwdrift', seasonal = 1).fit(method = 'powell', disp = False)
        return self
    
    def predict(self, h, CI):
        
        forecast_df = self.rwalkds.get_forecast(h).summary_frame(alpha = 0.05)
        y_hat = forecast_df['mean'].tolist()
        lower_b = forecast_df['mean_ci_lower'].tolist()
        upper_b = forecast_df['mean_ci_upper'].tolist()
        forecast_int_fin = [[x, y] for (x,y) in zip(lower_b, upper_b)]
        
        if CI:
            return forecast_int_fin
        else:
            return y_hat    

    
class LLevel:
    """
    Local Level
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.llevels = sm.tsa.UnobservedComponents(ts_init[-48:], level = 'llevel').fit(method = 'powell', disp = False)
        return self
    
    def predict(self, h, CI):
        forecast_df = self.llevels.get_forecast(h).summary_frame(alpha = 0.05)
        y_hat = forecast_df['mean'].tolist()
        lower_b = forecast_df['mean_ci_lower'].tolist()
        upper_b = forecast_df['mean_ci_upper'].tolist()
        forecast_int_fin = [[x, y] for (x,y) in zip(lower_b, upper_b)]
        
        if CI:
            return forecast_int_fin
        else:
            return y_hat        
    



class AutoArima:
    """
    AUTOARIMA
    """
    def __init__(self):
        pass
    
    def fit(self, ts_init, frcy):
        self.frcy = frcy
        self.arima = pm.auto_arima(
            ts_init[-36:],
            m=self.frcy,
            d=1,
            D=1,
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True
        )
        
        return self
    
    def predict(self, h, CI):
        
        y_hat, return_conf_int = self.arima.predict(h, return_conf_int = True, alpha = 0.05)

        if CI:
            return return_conf_int
        else:
            return y_hat
    

import tqdm
import numpy as np
import copy

class trainBasicModels:
    
        """
        Helper class that trains the models and returns predictions. 
        """
    
    def __init__(self):
        pass
    
    # Train functions
    def train_basic(self, model, ts, frcy):
        this_model = copy.deepcopy(model)
        if 'frcy' in model.fit.__code__.co_varnames:
            fitted_model = this_model.fit(ts, frcy)
        else:
            fitted_model = this_model.fit(ts)

        return fitted_model

    def train(self, basic_models, y_train_df,  frcy):
        
        """
        basic_models: Dict of models with name
        """
        
        ts_list = y_train_df.copy()
        
        self.model_names = basic_models.keys()
        self.basic_models_list = basic_models.values()
        self.fitted_models = [
            np.array([self.train_basic(model, ts, frcy) for model in self.basic_models_list]) for ts in tqdm.tqdm(ts_list)
        ]
        
        return self

    def predict(self, y_hat_df, CI):
                
        y_hat = [
            np.array([model.predict(y_hat_df, CI) for model in idts]) for idts in tqdm.tqdm(self.fitted_models)]
        
        return y_hat 
