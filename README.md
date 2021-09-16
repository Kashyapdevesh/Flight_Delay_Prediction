## Flight_Delay_Prediction

# Usage of the API  

This example considers that the API was launched locally without docker and 
with the default parameters (`localhost` at port `5003`) and its calling 
the example model.

# Docker commands 
Note: Docker tag or id should be always specified in the end of the docker command to avoid issues
1. Build docker image from Dockerfile

    ```docker build -t "<app name>" -f ./Dockerfile .```
    ```eg: docker build -t "ml_app" -f ./Dockerfile .```

2. Run the docker container after build

    ```docker run -p 5003 ml_app  # -p to make the port externally avaiable for browsers```

3. Show all running containers
    
    ```docker ps```

    a. Kill and remove running container
    
     ```docker rm <containerid> -f ```

4. Open bash in a running docker container (optional)

    ```docker exec -ti <containerid> bash```
5. Docker Entry point
The ENTRYPOINT specifies a command that will always be executed when the container starts. The CMD specifies arguments that will be fed to the ENTRYPOINT
1683

Docker has a default ENTRYPOINT which is /bin/sh -c but does not have a default CMD.
--entrypoint in docker run will overwrite the default entry point
    ```docker run -it --entrypoint /bin/bash <image>```


### Check the API's health status

Endpoint: `/health`

```bash
$ curl -X GET http://localhost:5000/health
up
```

### Wiping models after use (NOTICE: Future Scope for multi- model approach)

Endpoint: `/wipe`

```bash
$ curl -X GET http://localhost:5000/wipe
Model wiped
```

While all the above API requests can be accesed via simple HTTP 'GET' request, the prediction model requires input in JSON format accesible through 'POST' request as shown below :

```bash
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

```bash
{
    "Prediction": "45",
    "Prediction": "12"
}

```
Moreover, we have also provided provison for bulk predictions thorugh the same JSON requests.

## The model

#### The data 
![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/the%20data.png)


The independent variables :- Temperature, dew point, humidity, wind speed, precipitation, month, hours,The Weather delay(In minutes) or the target column which  we have to predict. 
The relationship between the variables is non-linear.
Temperature : the temperature at that time and day of the year.
Humidity : the Humidity at that time and day of the year.
Dew point: the Dew point at that time and day of the year.
Wind speed  : the Wind speed at that time and day of the year.
Precipitation : the Precipitation(%) at that time and day of the year.
Pressure : The pressure at that time and day of the year.
Month and hours : The month (in numerical form) and hours (24 hour format) 

From the above plot, we get a clear idea that linear models such as linear regression will not perform well on our given case.
We started with importing the necessary libraries: 
 

A brief about the model we employed :-
    Our approach involves supervised regression models :
    1) Lasso :-In this approach, we employed a grid search for the best Value for Lambda( 入 ) and S.

In this approach, we calculated the   method of lasso regression to find the best suited values of βi which are the coefficients for the independent variables in the model 

Ŷ =  Σ βi. Xi + c  where we reduce the RSS(Residual sum of squares) 

RSS = RSS + λ Σ |βi | where Σ|βi| <= S

For this purpose, we used GridSearchCV for finding the best value of alpha in python :
![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/lasso.png)

![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/lasso2.png)


    2) Random Forest : The next model we used was random forest regressor, which uses 

we could calculate
f ˆ 1 (x), f ˆ 2 (x), . . . , f ˆ B (x) using B separate training sets, and average them
in order to obtain a single low-variance statistical learning model, given by: 


        f ˆ avg (x) = 1/B (  Σ f^b(x) )
This method gives a less biased and low variance result with the use of the ensembling method.
In our code we employed it as follows: 

### rfc = RandomForestRegressor(n_estimators=200 , max_depth=15)


    3) XGboost Regressor : The next approach, we used was similar to the random forest approach, in the Extreme gradient boosting regression we make various decision trees for the purpose of predictions and the learning rate is decided based upon the descent, it adjusts itself with every step. 
We used the GridSearchCV for finding the best learning rate for the algo to work hence we employed the following code in python : 

  ![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/xgbregressor.png)
  ![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/xgbregressor2.png)


    4) Gaussian mixture networks : The last algorithm we used was gaussian mixture regression which is widely used for Multivariate Nonparametric regression problems such as ours, in this we use the probabilistic approach rather than direct values prediction method . We define the model as : 
     m(x) = E[Y |X =x ] (expected value of Y given  X= x) 
        =   Σ wj (x) . mj(x)
In python, we used the minimum error approach for finding the optimal cluster count
![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/gaussian_mixture.png)
![](https://github.com/shaurysrivastav27/Flight_Delay_Prediction/blob/main/notebooks/Final/plots/plot.jpg)
 
    
Finally, we had  to move beyond linearity to get our solution. We used the method of ensembling / stacking to get our best results using the StackingCVRegressor


The final model we used was a combination of all the models and the final prediction was the average of all the predicted weather delays. 
The final model had an mean absoulte error of 46 mins.


The final code which we used was : 


```bash
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







