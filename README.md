# Train_test_predict-of-ConvLSTM-in-Altimetry-data
This package contains the code used in the academic paper "Sea Level Prediction in the Kuroshio Extension Region Based on ConvLSTM".
train.py is the training code. test.py is the test code. predict.py is the prediction code, which is used to generate the prediction set.

Among them, predict.py uses autoregressive prediction, and the specific scheme is as follows:
Using observational data spanning from January 1st to December 31, 2015 (involving 365 time steps for prediction arrays), the SLA on January 1st, 2016 was predicted, and then observed SLA from January 2rd to December 31 in 2015 and predicted SLA on January 1st, 2016 were used as new "observed data" to predict SLA on January 2rd, 2016, and so on.
