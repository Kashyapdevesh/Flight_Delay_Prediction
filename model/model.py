import pandas as pd
import numpy as np

import joblib
import time 

df2016 = pd.read_csv('/home/shaury/Desktop/pvsc/alibaba/final.csv')
df2017 = pd.read_csv('/home/shaury/final2017.csv')
df2018 = pd.read_csv('/home/shaury/Desktop/pvsc/alibaba/final2016.csv')

df = pd.concat([df2016,df2017,df2018]).drop("Unnamed: 0",axis=1).reset_index(drop=True)

df['time'] = pd.to_datetime(df['time'],format= "%Y-%m-%d %H:%M:%S")
df['month'] = df['time'].dt.month
df['hours'] = df['time'].dt.hour
df['year'] = df['time'].dt.year

findf = df[(df['WEATHER_DELAY'].isna()==False) & (df['WEATHER_DELAY']>0)]

findf['month'] = pd.Categorical(findf['month'])
findf['Wind Speed'] = pd.Categorical(findf['Wind Speed'])
findf['Precipitation'] = pd.Categorical(findf['Precipitation'])

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(
    findf[['Temperature','Dew Point','Humidity','Wind Speed','Pressure','Precipitation','month']]
    ,findf['WEATHER_DELAY'],test_size=0.2)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(xtrain)
x_train = scaler.transform(xtrain)
x_test = scaler.transform(xtest)

from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor

def save_model(model,modelstr):
    to_save = dict(model=model,metadata={})
    model_path = '/home/shaury/Desktop/pvsc/alibaba/'+modelstr+'.joblib'
    with open(model_path, 'wb') as fo:
        joblib.dump(to_save, fo)
        
def main_model_function(x_train,ytrain,x_test,ytest):
    start = time.time()
    print(start)
    #gaussian mixture model
    err,I = 1000,0
    for i in range(1,40):
        gbmodel = GaussianMixture(n_components=i)
        gbmodel.fit(x_train,ytrain)
        if(err>mean_absolute_error(gbmodel.predict(x_test),ytest)):
            err = mean_absolute_error(gbmodel.predict(x_test),ytest)
            I = i
    gbmodel = GaussianMixture(n_components=I)
    gbmodel.fit(x_train,ytrain)
    save_model(gbmodel,"gbmodel")
    gbpred = gbmodel.predict(x_test)
    
    # lasso model
    alphas = 10**np.arange(-7,0,0.1)
    params = {"alpha":alphas}
    lassocv = GridSearchCV(Lasso(max_iter=1e7),param_grid=params,verbose = 5)
    lassocv.fit(x_train,ytrain)
    lassomodel = Lasso(alpha = lassocv.best_params_['alpha'],max_iter=1e7)
    lassomodel.fit(x_train,ytrain)
    save_model(lassomodel,"lassomodel")
    laspred = lassomodel.predict(x_test)
    
    # xgb random forest regression
    xgrmodel = XGBRFRegressor(gamma=10)
    xgrmodel.fit(x_train,ytrain)
    xgrpred = xgrmodel.predict(x_test)
    save_model(xgrmodel,"xgrmodel")
    
    #random forest regressor
    rfc = RandomForestRegressor()
    rfc.fit(x_train,ytrain)
    rfcpred = rfc.predict(x_test)
    save_model(rfc,"rfcmodel")
    
    #xgb regressor
    lrate = 10**(np.arange(-3,0.2,0.01))
    cvxg = GridSearchCV(XGBRegressor(n_estimators=150),param_grid={"learning_rate":lrate},verbose=5).fit(x_train,ytrain)
    xgbmodel = XGBRegressor(n_estimators=150,learning_rate=cvxg.best_params_['learning_rate'])
    xgbmodel.fit(x_train,ytrain)
    xgbpred = xgbmodel.predict(x_test)
    save_model(xgbmodel,"xgbmodel")
    
    stack = StackingCVRegressor(regressors=(gbmodel, lassomodel, rfc, xgrmodel, xgbmodel),
                            meta_regressor=xgbmodel, cv=10,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)
    stack.fit(x_train,ytrain)
    save_model(stack,"model")
    print('Trained in %.1f seconds' % (time.time()  -start))
    return stack.predict(x_test)
    
ypred = main_model_function(x_train,ytrain,x_test,ytest)

model_columns_file_name = './model_columns.joblib'
model_columns = list(x_test.columns)
joblib.dump(model_columns, model_columns_file_name)


