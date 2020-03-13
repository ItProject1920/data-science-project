#open anaconda base environment and app.py in it
import os
from flask import Flask, flash, request, render_template, url_for, redirect
#from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from werkzeug.utils import secure_filename
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn import neighbors

app = Flask(__name__, template_folder='templates')
#app.config['SQLALCHEMY_DATABASE_URL'] = 'sqlite:///test.db'
#rom flask_sqlalchemy import SQLAlchemydb = SQLAlchemy(app)

 

@app.route("/")
def fileFrontPage():
    return render_template('index.html')



@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '': 
            path = os.path.join('Upload/', '1.csv')
            photo.save(path)
    return redirect(url_for('fileFrontPage'))

@app.route("/suggest")
def suggest():
    return render_template('display.html')
	
@app.route("/display")
def Display():
    df = pd.read_csv('Upload/1.csv', sep=',')
    sf = df.head(5)
    i = list(df.columns.values)
    return render_template('display.html', tables=[sf.to_html()], index=i)

@app.route('/history')
def history():
    return render_template('history.html') 


@app.route('/comparison', methods=['GET', 'POST'])
def comparison():
    if request.method == "POST":
        selected_column = request.form.getlist("columnn")

    if request.method == "POST":
        selected_predict = request.form.getlist("predict")

    df = pd.read_csv('Upload/1.csv', sep=',')

    
    x = pd.DataFrame(df.iloc[:,:-1])
    y = pd.DataFrame(df.iloc[:,-1])


    
    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('main2.html')
    
    if request.method == 'POST':
        # Extract the input
        return render_template('main2.html',)

                                    
@app.route('/xgboost', methods=['GET', 'POST'])
def xgboost():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('main2.html')
    
    if request.method == 'POST':
        # Extract the input
        
        classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = classifier.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )    		


@app.route('/linearR', methods=['GET', 'POST'])
def linearR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('main2.html')
    
    if request.method == 'POST':
        # Extract the input
        classifier=LinearRegression()

        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = classifier.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )          

@app.route('/decisiontreeR', methods=['GET', 'POST'])
def decisiontreeR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('main2.html')
    
    if request.method == 'POST':
        # Extract the input
        classifier=DecisionTreeRegressor(max_depth=5,random_state=0)

        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = classifier.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )   

@app.route('/ridgeR', methods=['GET', 'POST'])
def ridgeR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('main2.html')
    
    if request.method == 'POST':
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
        
        return render_template('main2.html',
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
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('main2.html')
    
    if request.method == 'POST':
        knn=neighbors.KNeighborsRegressor(5,weights='distance')
        knn.fit(X_train,y_train)


        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = knn.predict(X_test)
        mae=mean_absolute_error(y_true=y_test, y_pred=predictions)
        evs=explained_variance_score(y_true=y_test, y_pred=predictions)
        acc=r2_score(y_true=y_test, y_pred=predictions)
        
        return render_template('main2.html',
                                     result=acc,
                                     result1=mae,
                                     result2=evs,
                                     )                                            




if __name__ == '__main__':
    app.run(debug=True)     
