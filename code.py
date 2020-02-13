# --------------
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
## Load the data
sales_data = pd.read_csv(path)

## Split the data and preprocess
#Divide into test and train:
train = sales_data.loc[sales_data['source']=="train"]
test = sales_data.loc[sales_data['source']=="test"]

#train = sales_data[sales_data['source']=="train"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'], axis=1, inplace=True)
train.drop(['source'],axis=1, inplace=True)

## Baseline regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X1 = train.loc[:, ['Item_Weight', 'Item_MRP', 'Item_Visibility']]

x_train1, x_cv1, y_train1, y_cv1 = train_test_split(X1, train.Item_Outlet_Sales, test_size=0.3, random_state =43)

# Intiating baseline model
alg1 = LinearRegression(normalize=True)
alg1.fit(x_train1, y_train1)

# Predicting on the sample subset 
yhat1 = alg1.predict(x_cv1)

# Calculating error
mse1 = np.mean((yhat1-y_cv1)**2)
print('Mean Squared Error is',mse1)

# R-Square
print('R Square Score is ', r2_score(y_cv1, yhat1))

## Effect on R-square if you increase the number of predictors
# Let's try out to set up a baseline model with just two explanatory variables
X2 = train.drop(columns=['Item_Outlet_Sales','Item_Identifier'])

x_train2, x_cv2, y_train2, y_cv2 = train_test_split(X2, train.Item_Outlet_Sales, test_size =0.3, random_state =100)

# Intiating baseline model
alg2 = LinearRegression(normalize=True)
alg2.fit(x_train2, y_train2)

# Predicting on the sample subset 
yhat2 = alg2.predict(x_cv2)

# Calculating error
mse2 = np.mean((yhat2-y_cv2)**2)
print('Mean Squared Error is',mse2)

## Effect of decreasing feature from the previous model
X3 = train.drop(columns=['Item_Outlet_Sales','Item_Identifier', 'Item_Visibility', 'Outlet_Years'])
x_train3, x_cv3, y_train3, y_cv3 = train_test_split(X3, train.Item_Outlet_Sales, test_size =0.3, random_state =100)

# Intiating baseline model
alg3 = LinearRegression(normalize=True)
alg3.fit(x_train3, y_train3)

# Predicting on the sample subset 
yhat3 = alg3.predict(x_cv3)

# Calculating error
mse3 = np.mean((yhat3-y_cv3)**2)
print('Mean Squared Error is',mse3)

# Implementing adjusted r square 
def adj_r2_score(model,y,yhat):
    from sklearn import metrics
    adj = 1 - float(len(y)-1)/(len(y)-len(model.coef_)-1)*(1 - metrics.r2_score(y,yhat))
    return adj

# Comparing r square and adjusted r square across three models
adj_score_model1 = adj_r2_score(alg1, y_cv1, yhat1)
adj_score_model2 = adj_r2_score(alg2, y_cv2, yhat2)
adj_score_model3 = adj_r2_score(alg3, y_cv3, yhat3)

print('R square {} and adjusted R square {} of model 1 '.format(r2_score(y_cv1, yhat1), adj_score_model1))
print('R square {} and adjusted R square {} of model 2 '.format(r2_score(y_cv2, yhat2), adj_score_model2))
print('R square {} and adjusted R square {} of model 3 '.format(r2_score(y_cv3, yhat3), adj_score_model3))

## Detecting hetroskedacity
# Lets plot residuals of model 2 
x_plot = plt.scatter(yhat2, (yhat2 - y_cv2), c='b')

plt.hlines(y=0, xmin= -1000, xmax=5000)

plt.title('Residual plot')

## Model coefficients
predictors = x_train2.columns

coef = pd.Series(alg2.coef_,predictors).sort_values()

plt.figure(figsize=(10,10))
coef.plot(kind='bar', title='Modal Coefficients')

## Ridge regression
from sklearn.linear_model import Ridge, Lasso

## training the model
def regularization_ridge(alpha):
    ridgeReg = Ridge(alpha=alpha, normalize=True)

    ridgeReg.fit(x_train2,y_train2)

    yhat_ridge = ridgeReg.predict(x_cv2)
    
    return yhat_ridge, ridgeReg

alpha_vals = [0.01, 0.05, 0.5, 5 ,10, 15, 25]
predictors = x_train2.columns

for i in alpha_vals:
    
    pred_ridge, model_ridge = regularization_ridge(i)
    coef = pd.Series(model_ridge.coef_,predictors).sort_values()

    plt.figure(figsize=(10,10))
    coef.plot(kind='bar', title='alpha {}'.format(i))
    
    print('R square for alpha value {} is {}'.format(i,r2_score(y_cv2, pred_ridge)))

## Lasso regression
alpha_vals_lasso = [0.01, 0.05, 0.5, 5]

def regularization_Lasso(alpha):
    lassoReg = Lasso(alpha=alpha, normalize=True)

    lassoReg.fit(x_train2,y_train2)

    yhat_lasso = lassoReg.predict(x_cv2)
    
    return yhat_lasso, lassoReg

for i in alpha_vals_lasso:
    
    pred_lasso, model_lasso = regularization_Lasso(i)
    coef = pd.Series(model_lasso.coef_,predictors).sort_values()

    plt.figure(figsize=(10,10))
    coef.plot(kind='bar', title='alpha {}'.format(i))
    
    print('R square for alpha value {} is {}'.format(i,r2_score(y_cv2, pred_lasso)))

## Cross vallidation
from sklearn import model_selection
Y = train['Item_Outlet_Sales']
X = train.drop(columns=['Item_Outlet_Sales','Item_Identifier'])
kfold = model_selection.KFold(n_splits=10, random_state=100)

results = model_selection.cross_val_score(Ridge(0.01, normalize=True), X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))



