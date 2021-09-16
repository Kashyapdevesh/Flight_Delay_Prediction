# Flight Delay Prediction API

The API in this repository is our team's submission for Alibaba Global AI Challenge 2021 finals.The theme of this challenge was, "Intelligent Weather Forecast for Better Life" and we were required to propose state of the art solutions pertaining to the theme.



In this project we developed a high level customizable API which could be used to predict flight delay on any airport around the world. Initially, the model trained itself by scraping past weather data and flight details of the given airport. Then, it containerized itself and set up a production grade flask server incorporated with nginx and gunicorn which is further deployed to Alibaba's cloud service and published in **Alibaba's API marketplace**.



Morever, the process mentioned above is automated with the use of Alibaba workflow in this github repository, so that during the official release the changes could easily be reflected in the production API without any delay or overhead.



During construction of this project, the main focus was on flexibility, so that it could easily
be used by professionals and beginners alike. Moreover, we have tried to keep the API as
user-friendly as possible while maintaining its robustness.

The developed API can later be used for a multitude of purposes, such as:

1. Creating a mobile/web app which depicts flight weather delays to customers with a very high accuracy
2. Integration with airline maintenance systems to predict optimum time for flight maintenance
3. Integration with Air Traffic Control (ATC) for better management
4. Integration with airline booking system to increase efficiency

Last but not the least, our model is better than existing options as it’s easily scalable, not
much interference is required after initial setup and it gives almost accurate prediction for
delayed flights.

Table of contents
=================

<!--ts-->
   * [About the predictive model](#About-the-predictive-model)
   * [Sample training data](#Sample-training-data)
   * [Model Parmaters'](#Model-Parmaters)
   * [Model Methodology](#Model-Methodology)
       * [Lasso](#Lasso)
       * [Random Forest](#Random-Forest)
       * [XGboost Regressor](#XGboost-Regressor)
       * [Gaussian mixture networks](#Gaussian-mixture-networks)
       * [Final Ensembled Model](#Final-Ensembled-Model)
   * [Usage of the API](#Usage-of-the-API)
   * [Docker Commands](#Docker-Commands)
   * [Check the API's health status](#Check-the-API-health-status)
   * [Wiping models after use](#Wiping-models-after-use)
   * [Alibaba products' used](#Alibaba-products-used)
<!--te-->

## About the predictive model

## Sample training data
![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/the%20data.png)


## Model Parmaters'


The independent variables :- **Temperature**, **dew point**, **humidity**, **wind speed**, **precipitation**, **month**, **hours**. 
The Weather delay(in minutes) or the target column is the one we have to predict. 
The relationship between the variables is non-linear.
1. **Temperature** : the temperature at that time and day of the year.
2. **Humidity** : the Humidity at that time and day of the year.
3. **Dew point**: the Dew point at that time and day of the year.
4. **Wind speed** : the Wind speed at that time and day of the year.
5. **Precipitation** : the Precipitation(%) at that time and day of the year.
6. **Pressure** : The pressure at that time and day of the year.
7. **Month and hours** : The month (in numerical form) and hours (24 hour format) 


## Model Methodology
For our final model we used an ensemble of various models listed below individually:

The complete model is present in https://github.com/Kashyapdevesh/Flight_Delay_Prediction/blob/main/notebooks/Final/Final%20model.ipynb 

## Lasso: 
In this approach, we employed a grid search for the best Value for Lambda( 入 ) and S.
Initially, we calculated the method of lasso regression to find the best suited values of βi which are the coefficients for the independent variables in the model 
```python
Ŷ =  Σ βi. Xi + c  where we reduce the RSS(Residual sum of squares) 
```
```python
RSS = RSS + λ Σ |βi | where Σ|βi| <= S
```
For this purpose, we used GridSearchCV for finding the best value of alpha in python :
```python 
#Lasso model
alphas =10**np.arrange(-7,0,0.1)
params={"alpha":alphas}
lassocv=GridSearchCV(Lasso(max_iter=1e7),param_grif=params,verbose=5)
lassocv.fit(x_train,ytrain)
lassomodel=Lasso(alpha=lassocv.best_param_['alpha'],max_iter=1e7)
```

![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/lasso2.png)

## Random Forest: 
The next model we used was random forest regressor, which uses 

we could calculate
f ˆ 1 (x), f ˆ 2 (x), . . . , f ˆ B (x) using B separate training sets, and average them
in order to obtain a single low-variance statistical learning model, given by: 

```python
f ˆ avg (x) = 1/B (  Σ f^b(x))
```         
        
This method gives a less biased and low variance result with the use of the ensembling method.
In our code we employed it as follows: 

```python
rfc = RandomForestRegressor(n_estimators=200 , max_depth=15)
```

## XGboost Regressor: 
The next approach, we used was similar to the random forest approach, in the Extreme gradient boosting regression we make various decision trees for the purpose of predictions and the learning rate is decided based upon the descent, it adjusts itself with every step. 
We used the GridSearchCV for finding the best learning rate for the algo to work hence we employed the following code in python : 

```python
#xgb regressor
lrate=10**(np.arrange(-2,0.2,0.01))
cvxg =GridSearchCV(XGBRegressor(n_estimators=150),param_grid={"learning_rate":lrate},verbose=5).fit(x_train,ytrain))    
 xgmodel=XGBRegressor(n_estimators=150,learning_rate=cvxg.best_params_['learning_rate'])
 ```
  ![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/xgbregressor2.png)

## Gaussian mixture networks: 

The last algorithm we used was gaussian mixture regression which is widely used for Multivariate Nonparametric regression problems such as the given problem.
In this approach, we use the probabilistic approach rather than direct values prediction method . We define the model as : 
```python
     m(x) = E[Y |X =x ] (expected value of Y given  X= x) 
        =   Σ wj (x) . mj(x)
```
In python, we used the minimum error approach for finding the optimal cluster count
```python
#gaussian mixture model
err,I=1000,0
for i in range(1,40):
    gbmodel= GaussianMixture(n_components=i)
    gbmodel.fit(x_train,ytrain)
    if(err>mean_absolute_error(gbmodel.predict(x_train),y_train)):
        err=mean_absolute_error(gbmodel.predict(x_train),y_train)
        I=i
 gbmodel= GaussianMixture(n_components=I)
 ```
 
![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/plot.jpg)
 
## Final Ensembled Model
 
Finally, we had  to move beyond linearity to get our solution. We used the method of ensembling / stacking to get our best results using the StackingCVRegressor


The final model we used was a combination of all the models and the final prediction was the average of all the predicted weather delays. 
The final model had an mean absoulte error of 46 mins.

The final code which we used was : 


```python
{
    def main_model_function(x_train,ytrain):
    start = time.time()
    #gaussian mixture model
    err,I = 1000,0
    for i in range(1,40):
        gbmodel = GaussianMixture(n_components=i)
        gbmodel.fit(x_train,ytrain)
        if(err>mean_absolute_error(gbmodel.predict(x_train),ytrain)):
            err = mean_absolute_error(gbmodel.predict(x_train),ytrain)
            I = i
    gbmodel = GaussianMixture(n_components=I)
    # lasso model
    alphas = 10**np.arange(-7,0,0.1)
    params = {"alpha":alphas}
    lassocv = GridSearchCV(Lasso(max_iter=1e7),
                             param_grid=params,verbose = 5)
    lassocv.fit(x_train,ytrain)
    lassomodel = Lasso(alpha = lassocv.best_params_['alpha'],max_iter=1e7)
    
    
    #random forest regressor
    rfc = RandomForestRegressor(n_estimators=200 , max_depth=15)
    
    
    #xgb regressor
    lrate = 10**(np.arange(-2,0.2,0.01))
    cvxg = GridSearchCV(XGBRegressor(n_estimators=150),param_grid={"learning_rate":lrate},verbose=5).fit(x_train,ytrain)
    xgbmodel = XGBRegressor( n_estimators=150,
                learning_rate=cvxg.best_params_['learning_rate'])
      stack = StackingCVRegressor(regressors=
              (gbmodel, lassomodel, rfc, xgbmodel),
                              meta_regressor=xgbmodel, cv=10,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)
    stack.fit(x_train,ytrain)
    print(start- time.time())
    return stack

}

```

## Usage of the API  

This example considers that the API was launched locally without docker and 
with the default parameters (`localhost` at port `5003`) and its calling 
the example model.

## Docker Commands 
Note: Docker tag or id should be always specified in the end of the docker command to avoid issues

1. Build docker image from Dockerfile

    ```bash 
    sudo docker build -t "<app name>" -f ./Dockerfile .
    ```
    ```bash
    eg: sudo docker build -t "ml_app" -f ./Dockerfile .
    ```


    1. If you don't want to manually build the Docker Image, you can pull the docker image from **Alibaba Container Registry** using the commands below:
        ```bash
        sudo docker pull registry-intl.ap-south-1.aliyuncs.com/flight_delay_prediction_ml/flight_delay_prediction:latest
        ```   
        
    2. You can also pull the docker image from dockerhub using the command below:
        ```bash
        sudo docker pull kashyapdevesh2412/flight_delay_prediction
        ```

2. Run the docker container after build

```bash
sudo docker run -p 5003 ml_app  # -p to make the port externally avaiable for browsers
```

3. Show all running containers
    
```bash
sudo docker ps
```

4. Kill and remove running container
    
```bash
sudo docker rm <containerid> -f
```

5. Open bash in a running docker container (optional)

```bash
sudo docker exec -ti <containerid> bash
```

6. Docker Entry point
The ENTRYPOINT specifies a command that will always be executed when the container starts. The CMD specifies arguments that will be fed to the ENTRYPOINT
1683

Docker has a default ENTRYPOINT which is /bin/sh -c but does not have a default CMD.
--entrypoint in docker run will overwrite the default entry point
```bash
sudo docker run -it --entrypoint /bin/bash <image>
```


## Check the API health status

Endpoint: `/health`

```bash
$ curl -X GET http://localhost:5000/health
up
```

## Wiping models after use 

**NOTE: Future Scope for multi- model approach**

This feature was added in case of expansion and integration of the API with other services.
**NOTE:Don't use this feature during local testing as it can wipe the current model in use**

Endpoint: `/wipe`

```bash
$ curl -X GET http://localhost:5000/wipe
Model wiped
```

While all the above API requests can be accesed via simple HTTP 'GET' request, the prediction model requires input in JSON format accesible through 'POST' request as shown below :

```json
[
    {
        "iataCode": "ATL",
        "date": "2021-09-12",
        "time": "16:25"
    },
    {
        "iataCode": "JFK",
        "date": "2021-09-13",
        "time": "12:30"
    }
]


```
The API will provide output prediction in the same JSON format as shown below:

```json
{
    "Prediction": "45",
    "Prediction": "12"
}

```
Moreover, we have also provided provison for bulk predictions thorugh the same JSON requests.

## Alibaba products used:
1. **Alibaba Container Registory**: For conatinerizing and hosting our final docker image
2. **Alibaba Kubernetes Cluster** : Used in our repository's workflow for continous integration and deployment
3. **Alibaba Elastic Compute Server**: For hosting and deploying our API on independent apache server
4. **Alibaba API gateway**:Used to publish our API in Alibaba's API marketplace




