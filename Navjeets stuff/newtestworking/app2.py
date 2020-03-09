import flask
from flask import Flask, request, render_template, url_for, redirect
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn import neighbors



from sklearn.datasets import load_boston
boston=load_boston()

data=pd.DataFrame(boston.data)
data.columns=boston.feature_names


x = data[['LSTAT', 'RM', 'RAD']]
y = data['TAX']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)




# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
		

		
        return flask.render_template('main2.html',
                                     )
                                     
@app.route('/xgboost', methods=['GET', 'POST'])
def xgboost():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        
        classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = classifier.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )    		


@app.route('/linearR', methods=['GET', 'POST'])
def linearR():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        classifier=LinearRegression()

        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = classifier.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )          

@app.route('/decisiontreeR', methods=['GET', 'POST'])
def decisiontreeR():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        classifier=DecisionTreeRegressor(max_depth=5,random_state=0)

        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = classifier.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )   

@app.route('/ridgeR', methods=['GET', 'POST'])
def ridgeR():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
        ridge = Ridge()
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}
        ridge_reg = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

        #ridge_reg = Ridge(alpha=0.01, solver="cholesky")
        ridge_reg.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = ridge_reg.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )

@app.route('/lassoR', methods=['GET', 'POST'])
def lassoR():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}
        lasso=Lasso()
        lassoReg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)
        #lassoReg = Lasso(alpha=0.0001,normalize=True)
        lassoReg.fit(X_train,y_train)


        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = lassoReg.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     ) 

@app.route('/knnR', methods=['GET', 'POST'])
def knnR():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        knn=neighbors.KNeighborsRegressor(5,weights='distance')
        knn.fit(X_train,y_train)


        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = knn.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )                                            


if __name__ == '__main__':
    app.run(debug=True)