# Thesis code ESE Jasper Van den Neste
## "Forecasting the market development of the automotive industry using Feature Based Forecast Model Averaging"

This research aims at developing a methodology for forecasting the market development for the automotive industry for up to three years in the future. More accurately, the focus lies on finding accurate ways to proxy the market development, and then using a feature-based forecast model averaging approach to forecast this proxied market development.

This document describes how the scripts can be used for creating the Vehicle Price Index and for forecasting it. For further explanation of the logic behind this method, we refer to the document *Forecasting Car Price Index with Feature Based Forecast Model Averaging (Thesis)*. Note, the results in the scripts can differ slightly from the results in the thesis, due to the use of different pretrained meta-learning models.


### Creating the Index
The folder *Index methods* contains the script *Index methods.ipynb*. In the script, one has to determine the *Time_freq*, which denotes the frequency of the Index. Additionally, one has to determine which countries are of interest in the variable *countries*. Once this is done, the script automatically calculates the Index for with the *Bucket Average* method, the *Time Dummy* and the *Hedonic Imputation* method for all countries and saves the results in .pkl files. Note that the initial loading of the data was done via a Sagemaker instance through Amazon Web Services. Therefore, this part of the code was taken out of this file. Additionally, due to confidentiality, the files containing the Index  data are also left out.

### Training the FFORMA model
The script *Train_metalearning* can be used for training the FFORMA meta-learning model – both for the point forecasts as for the prediction intervals. For this script, a few packages are used/created. On the one hand, the package that describes the FFORMA meta-learning model itself. This can be found in the */fforma/* folder. This package is based on code from https://github.com/christophmark/fforma (previously https://github.com/FedericoGarza/fforma). However, this package was not fully created when we started this project, and therefore we created a working version ourselves. Next to that, our implementation is different than the approach in the original article by Montero-Manso et al (https://doi.org/10.1016/j.ijforecast.2019.02.011). Additionally, the folder */tsfeatures/* is used. This package calculates all time series features used in the FFORMA model and is an earlier version of the code found on https://github.com/FedericoGarza/tsfeatures.

The script *Train_metalearning* reads the datafiles *Monthly-train.csv* and *Monthly-test.csv*, which contain the time series from the M4 competition. It then loads the models in the model pool, which are described in the helper file *models.py*. For all time series in the datafiles that are selected, in then produces forecasts for all the models in the model pool. These forecasts are saved in pickle files named *./Results/Results_curr/Preds_set.pkl*. It then loads the predictions that are previously trained, and reformats the data. Additionally, it recalculates and loads the features corresponding to the time series.

It then trains the meta-learning models, both based on the point forecasts and the prediction intervals. Finally, the performance of the models based on a out-of-sample reference set is calculated and analyzed.

### Forecasting with FFORMA
This script produces the forecasts on a set of 'new' time series. For this, it uses the function *produce_PI_PF*. Predictions on a number of countries have already been done, and the results can be found in the script as well. The previously trained models and results are saved in the folders *Models* and *Results*.
