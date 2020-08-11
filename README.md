# Thesis code ESE Jasper Van den Neste
## "Forecasting the market development of the automotive industry using Feature Based Forecast Model Averaging"

This research aims at developing a methodology for forecasting the market development for the automotive industry for up to three years in the future. More accurately, the focus lies on finding accurate ways to proxy the market development, and then using a feature-based forecast model averaging approach to forecast this proxied market development.

This document describes how the scripts can be used for creating the Vehicle Price Index and for forecasting it. For further explanation of the logic behind this method, we refer to the document *Forecasting Car Price Index with Feature Based Forecast Model Averaging (Thesis)*.


### Creating the Index
The folder *Index methods* contains the script *Index methods.ipynb*. In the script, one has to determine the \textit{Time\_freq}, which denotes the frequency of the Index. Additionally, one has to determine which countries are of interest in the variable \textit{countries}. Once this is done, the script automatically calculates the Index for with the *Bucket Average* method, the *Time Dummy* and the *Hedonic Imputation* method for all countries and saves the results in .pkl files. Note that the initial loading of the data was done via a Sagemaker instance through Amazon Web Services. Therefore, this part of the code was taken out of this file.

### Training the FFORMA model
The script *Train_metalearning* can be used for training the FFORMA meta-learning model – both for the point forecasts as for the prediction intervals. For this script, a few packages are used/created. On the one hand, the package that describes the FFORMA meta-learning model itself. This can be found in the *Extra_packages/fforma/* folder. This package is based on code from https://github.com/christophmark/fforma (previously https://github.com/FedericoGarza/fforma). However, this was not fully created when we started this project, and therefore we created a working version ourselves. Additionally, the package *Extra_packages/tsfeatures/* is used. This package calculates all time series features used in the FFORMA model and is based on https://github.com/FedericoGarza/tsfeatures.
