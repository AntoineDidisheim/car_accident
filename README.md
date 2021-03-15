# car_accident
A simple data-science project in python. 

We first load and explore a data-set containing road accidents across the U.K., with the accident's location and conditions (weather, etc.). 

We produce a few descriptive statistics before turning to the primary analysis: 
1) we split the map of the U.K. into arbitrary regions using latitudes and longitudes. 
2) we define a few features, including average daily weather condition per region, to predict the number of accidents per day per region. 
3) we train three models: Linear Regression (OLS), Lasso, and Random Forests (RF), on an expanding window. 
4) we measure our models' performance out-of-sample and compare it against a simple benchmark---that is, the average number of accidents per region at time t-1. 
5) we show that the OLS overfit in-sample, the Lasso barely match the benchmark's performance, while the RF systematically outperforms our benchmark across time. 
6) we find that cross-sectional ---that is, region per region---the RF does not always outperform the benchmark. Specifically, we find that on regions with very few accidents per day, the RF systematically over-estimates the risk of an accident. 

#### Possible extensions: 
1) Code a grid search over the model's hyper-parameter to see if better model tuning can significantly improve performance out-of-sample
2) Improve predictive power by adding new features. 
3) Refine the region grid and see if the RF can still outperform the benchmark with more precise predictions
4) Redefine the region by taking into account the area code (urban/nonurban etc.) to create a model more valuable to potential decision-makers. 

## Code
* **parameters.py** contains the parameters, including saving and data path, and model's hyper-parameters and constant. The parameters also define the unique name under which the analysis result will be saved.
* **data.py** organize the data loading and data-pre-processing
* **map.py** contains the functions necessary to create graphs of the accidents' locations
* **data_exploration** look at the data structure and produces a few exploratory graphs. 
* **train_models.py** create the predicting features, train the models and save the performances out-of-sample. 
* **test_performance.py** load the output of *train_models.py*, compute performance across time and investigate the interesting patterns in performance that arrises. 


## Run order
1) *data_exploration.py*
2) *train_models.py*
3) *test_performances.py*