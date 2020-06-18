#open anaconda base environment and app.py in it
import os
from flask import Flask, flash, request, render_template, url_for, redirect
#from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import time
#Regression imports
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn import neighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
#clustering imports
from sklearn.cluster import *
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import *
from sklearn.mixture import GaussianMixture
from flask_jsglue import JSGlue
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from itertools import cycle, islice


app = Flask(__name__, template_folder='templates')
#app.config['SQLALCHEMY_DATABASE_URL'] = 'sqlite:///test.db'
#rom flask_sqlalchemy import SQLAlchemydb = SQLAlchemy(app)


jsglue = JSGlue(app)


@app.route("/")
def fileFrontPage():
    return render_template('index.html')

@app.route("/fileFrontPage2")
def fileFrontPage2():
    return render_template('preclustering2.html')

@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '': 
            path = os.path.join('Upload/', '1.csv')
            photo.save(path)
    return redirect(url_for('fileFrontPage'))

@app.route("/handleUpload2", methods=['POST'])
def handleFileUpload2():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '': 
            path = os.path.join('Upload/', '2.csv')
            photo.save(path)
    return redirect(url_for('fileFrontPage2'))
	
@app.route("/display")
def Display():
    df = pd.read_csv('Upload/1.csv', sep=',')
    sf = df.head(5)
    i = list(df.columns.values)
    ptr=sf.to_html().find('</thead>')
    return render_template('display_cal.html', tables1=[sf.to_html()[:ptr+9]], tables2= [sf.to_html()[ptr+9:]], index=i)

@app.route("/display_pre")
def Display_pre():
    df = pd.read_csv('Upload/1.csv', sep=',')
    sf = df.head(5)
    i = list(df.columns.values)
    ptr=sf.to_html().find('</thead>')
    return render_template('display_pre.html', tables1=[sf.to_html()[:ptr+9]], tables2= [sf.to_html()[ptr+9:]], index=i)

@app.route('/history')
def history():
    return render_template('history.html') 


@app.route('/comparison', methods=['GET', 'POST'])
def comparison():
    algorithms=['XGBoost Regression','Linear Regression','Decisiontree Regression', 'Ridge Regression', 'Lasso Regression','Elastic Net Regression', 'K-nn Regression','Support Vector Regression']
    
    
    if request.method == "POST":
        selected_column = request.form.getlist("column")

    if request.method == "POST":
        selected_predict = request.form.getlist("predict")

    df = pd.read_csv('Upload/1.csv', sep=',')


    x = df.loc[:,selected_column]
    y = df.loc[:,selected_predict]

    model = ExtraTreesRegressor()
    rfe = RFE(model, 3)
    fit = rfe.fit(x,y)

    print("Number of Features: ", fit.n_features_)
    print("Selected Features: ", fit.support_)
    print("Feature Ranking: ", fit.ranking_) 
    
    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    
    global mae,mse,rmse,evs,r2
    mae=[]
    mse=[]
    rmse=[]
    evs=[] 
    r2=[]
    
    matrix=[]
    matrix.append(xgboost())
    matrix.append(linearR())
    matrix.append(decisiontreeR())
    matrix.append(ridgeR())
    matrix.append(lassoR())
    matrix.append(elastNetR())
    matrix.append(knnR())
    matrix.append(svR())

    dataF=pd.DataFrame({
        'Algorithm':algorithms,
        'EVS':evs,
        'R2':r2,
        'MSE':mse,
        'MAE':mae,
        'RMSE':rmse    
        })
    #dataF['Rank'] = dataF.evs + dataF.r2 
    #dataF['Rank2'] = dataF.mse + dataF.mae+ dataF.rmse
    dataF.sort_values(by=['EVS'], inplace=True, ascending=False)
    dataF.sort_values(by=['R2'], inplace=True, ascending=False)
    dataF.sort_values(by=['MSE'], inplace=True, ascending=True)
    dataF.sort_values(by=['MAE'], inplace=True, ascending=True)
    dataF.sort_values(by=['RMSE'], inplace=True, ascending=True)

    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':

        return render_template('regression.html',sc=selected_column, algo= algorithms, mat=matrix,ranks=dataF)

                                    
@app.route('/xgboost', methods=['GET', 'POST'])
def xgboost():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        # Extract the input
        
        classifier = xg_reg = xgb.XGBRegressor()
        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = classifier.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
        mae1=mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse1=mean_squared_error(y_true=y_test, y_pred=predictions)
        rmse1=np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        rsquare=r2_score(y_true=y_test, y_pred=predictions)
        explainedv=explained_variance_score(y_true=y_test, y_pred=predictions)
        evs.append(explainedv)
        r2.append(rsquare)
        mae.append(mae1)
        mse.append(mse1)
        rmse.append(rmse1)
        return (responce)

@app.route('/linearR', methods=['GET', 'POST'])
def linearR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        # Extract the input
        classifier=LinearRegression()

        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = classifier.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
        mae1=mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse1=mean_squared_error(y_true=y_test, y_pred=predictions)
        rmse1=np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        rsquare=r2_score(y_true=y_test, y_pred=predictions)
        explainedv=explained_variance_score(y_true=y_test, y_pred=predictions)
        evs.append(explainedv)
        r2.append(rsquare)
        mae.append(mae1)
        mse.append(mse1)
        rmse.append(rmse1)
        return (responce)

@app.route('/decisiontreeR', methods=['GET', 'POST'])
def decisiontreeR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        # Extract the input
        classifier=DecisionTreeRegressor()

        classifier.fit(X_train, y_train)

        
        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = classifier.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
        mae1=mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse1=mean_squared_error(y_true=y_test, y_pred=predictions)
        rmse1=np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        rsquare=r2_score(y_true=y_test, y_pred=predictions)
        explainedv=explained_variance_score(y_true=y_test, y_pred=predictions)
        evs.append(explainedv)
        r2.append(rsquare)
        mae.append(mae1)
        mse.append(mse1)
        rmse.append(rmse1)
        return (responce)

@app.route('/ridgeR', methods=['GET', 'POST'])
def ridgeR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
        ridge = Ridge()
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}
        ridge_reg = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

        #ridge_reg = Ridge(alpha=0.01, solver="cholesky")
        ridge_reg.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = ridge_reg.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
        mae1=mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse1=mean_squared_error(y_true=y_test, y_pred=predictions)
        rmse1=np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        rsquare=r2_score(y_true=y_test, y_pred=predictions)
        explainedv=explained_variance_score(y_true=y_test, y_pred=predictions)
        evs.append(explainedv)
        r2.append(rsquare)
        mae.append(mae1)
        mse.append(mse1)
        rmse.append(rmse1)
        return (responce)

@app.route('/lassoR', methods=['GET', 'POST'])
def lassoR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('regression.html'))
    
    if request.method == 'POST':
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}
        lasso=Lasso()
        lassoReg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)
        #lassoReg = Lasso(alpha=0.0001,normalize=True)
        lassoReg.fit(X_train,y_train)


        
        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = lassoReg.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
        mae1=mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse1=mean_squared_error(y_true=y_test, y_pred=predictions)
        rmse1=np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        rsquare=r2_score(y_true=y_test, y_pred=predictions)
        explainedv=explained_variance_score(y_true=y_test, y_pred=predictions)
        evs.append(explainedv)
        r2.append(rsquare)
        mae.append(mae1)
        mse.append(mse1)
        rmse.append(rmse1)
        return (responce)



@app.route('/elastNetR', methods=['GET', 'POST'])
def elastNetR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('regression.html'))
    
    if request.method == 'POST':
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20],'l1_ratio':[0.1]}
        elasticNet=ElasticNet()
        elasNetReg = GridSearchCV(elasticNet, parameters, scoring='neg_mean_squared_error', cv = 5)
        # Fit/train LASSO
        elasNetReg.fit(X_train,y_train)
        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = elasNetReg.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
        mae1=mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse1=mean_squared_error(y_true=y_test, y_pred=predictions)
        rmse1=np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        rsquare=r2_score(y_true=y_test, y_pred=predictions)
        explainedv=explained_variance_score(y_true=y_test, y_pred=predictions)
        evs.append(explainedv)
        r2.append(rsquare)
        mae.append(mae1)
        mse.append(mse1)
        rmse.append(rmse1)
        return (responce)

@app.route('/knnR', methods=['GET', 'POST'])
def knnR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        knn=neighbors.KNeighborsRegressor(5,weights='distance')
        knn.fit(X_train,y_train)
        
        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = knn.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
        mae1=mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse1=mean_squared_error(y_true=y_test, y_pred=predictions)
        rmse1=np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        rsquare=r2_score(y_true=y_test, y_pred=predictions)
        explainedv=explained_variance_score(y_true=y_test, y_pred=predictions)
        evs.append(explainedv)
        r2.append(rsquare)
        mae.append(mae1)
        mse.append(mse1)
        rmse.append(rmse1)
        return (responce)

@app.route('/SVR', methods=['GET', 'POST'])
def svR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        regr = make_pipeline(StandardScaler(), SVR())
        regr.fit(X_train, y_train)
        
        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = regr.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
        mae1=mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse1=mean_squared_error(y_true=y_test, y_pred=predictions)
        rmse1=np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        rsquare=r2_score(y_true=y_test, y_pred=predictions)
        explainedv=explained_variance_score(y_true=y_test, y_pred=predictions)
        evs.append(explainedv)
        r2.append(rsquare)
        mae.append(mae1)
        mse.append(mse1)
        rmse.append(rmse1)
        return (responce)                                                        

@app.route('/prediction_classification', methods=['GET', 'POST'])
def Prediction_classification():
    if request.method == "POST":
        selected_column = request.form.getlist("column")

    if request.method == "POST":
        selected_predict = request.form.getlist("predict")

    df = pd.read_csv('Upload/1.csv', sep=',')
    algorithms=['Decision Tree','Randomforest','Logistic Regression', 'SVM Classifier', 'Knn Classifier', 'MLP Classifier', 'ADA Boost', 'Gaussian Naive Bayes', 'Quadratic Discriminant Analysis', 'MultinomialNB']

    x = df.loc[:,selected_column]
    y = df.loc[:,selected_predict]

    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    global ac,pr,re,f1
    ac=[]
    pr=[]
    re=[]
    f1=[]

    matrix=[]
    matrix.append(decisiontreeC())
    matrix.append(randomforestC())
    matrix.append(logisticR())
    matrix.append(svmC())
    matrix.append(knnC())
    matrix.append(mlpC())
    matrix.append(adC())
    matrix.append(nbC())
    matrix.append(qdaC())
    matrix.append(ngnbC())

    copy_matrix=[]
    ranker=[]
    copy_matrix=matrix

    for i in range(10):
        temp = copy_matrix[i][3]
        a=temp.at['accuracy','support']
        f=temp.at['weighted avg','f1-score']
        p=temp.at['weighted avg','precision']
        r=temp.at['weighted avg','recall']
        sum=a+f+p+r
        ranker.append(sum)
    e=np.array(matrix)
    f=np.array(ranker)
    final=np.column_stack((e,f))
    final = final[np.argsort(final[:, 4])]
    final=final[::-1]

    dataF=pd.DataFrame({
        'Algorithm':algorithms,
        'Accuracy':ac,
        'Precision':pr,
        'Recall':re,
        'FScore':f1
        })
    dataF['Rank'] = dataF.Accuracy + dataF.Precision + dataF.Recall + dataF.FScore
    dataF.sort_values(by=['Rank'], inplace=True, ascending=False)

    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('classification.html')
    
    if request.method == 'POST':
        # Extract the input
        return render_template('classification.html',sc=selected_column, mat=final,ranks=dataF)

@app.route('/decisiontreeC', methods=['GET', 'POST'])
def decisiontreeC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        # Extract the input
        #decision tree classifier
        from sklearn.tree import DecisionTreeClassifier
        decision_tree = DecisionTreeClassifier()
        start_time = time.time()
        decision_tree = decision_tree.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)
        
        #from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error
        
        predictions = decision_tree.predict(X_test)

        responce=[]
        responce.append('Decision Tree Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)        

@app.route('/randomforestC', methods=['GET', 'POST'])
def randomforestC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        # Extract the input
        random_forest = RandomForestClassifier()
        start_time = time.time()
        random_forest.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'Random Forest Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = random_forest.predict(X_test)

        responce=[]
        responce.append('Random Forest Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)      

@app.route('/logisticR', methods=['GET', 'POST'])
def logisticR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        # Extract the input
        logistic = LogisticRegression()
        start_time = time.time()
        logistic.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'Logistic Regression Classifier took {:.5f} s'.format(end_time - start_time)
        predictions = logistic.predict(X_test)
        responce=[]
        responce.append('Logistic Regression Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())
        
        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)      

@app.route('/svmC', methods=['GET', 'POST'])
def svmC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        # Extract the input
        support_vector = SVC()
        start_time = time.time()
        support_vector.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'SVM Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = support_vector.predict(X_test)        
        
        responce=[]
        responce.append('SVM Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)      

@app.route('/knnC', methods=['GET', 'POST'])
def knnC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        KNN = KNeighborsClassifier(n_neighbors=5)
        start_time = time.time()
        KNN.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'K-nn Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = KNN.predict(X_test)
        responce=[]
        responce.append('K-nn Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)      

@app.route('/gpC', methods=['GET', 'POST'])
def gpC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        gpC = GaussianProcessClassifier(1.0 * RBF(1.0))
        start_time = time.time()
        gpC.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'Gaussian Process Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = gpC.predict(X_test)
        acc=accuracy_score(y_test,predictions)
        cf=confusion_matrix(y_test,predictions)
        cr=classification_report(y_test,predictions, output_dict=True)       
        df = pd.DataFrame(cr).transpose()       

        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )

@app.route('/mlpC', methods=['GET', 'POST'])
def mlpC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        mlpC =MLPClassifier(alpha=1, max_iter=1000)
        start_time = time.time()
        mlpC.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'MLP Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = mlpC.predict(X_test)
        responce=[]
        responce.append('MLP Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)      

@app.route('/adC', methods=['GET', 'POST'])
def adC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        adC =AdaBoostClassifier()
        start_time = time.time()
        adC.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'Ada Boost Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = adC.predict(X_test)
        
        responce=[]
        responce.append('Ada Boost Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)      

@app.route('/nbC', methods=['GET', 'POST'])
def nbC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        nbC =GaussianNB()
        start_time = time.time()
        nbC.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'Gaussian Naive Bayes Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = nbC.predict(X_test)
        responce=[]
        responce.append('Gaussian Naive Bayes Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)

@app.route('/qdaC', methods=['GET', 'POST'])
def qdaC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        qdaC =QuadraticDiscriminantAnalysis()
        start_time = time.time()
        qdaC.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'Quadratic Discriminant Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = qdaC.predict(X_test)
        responce=[]
        responce.append('Quadratic Discriminant Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)

@app.route('/ngnbC', methods=['GET', 'POST'])
def ngnbC():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('classification.html'))
    
    if request.method == 'POST':
        ngnbC =MultinomialNB()
        start_time = time.time()
        ngnbC.fit(X_train,y_train)
        end_time = time.time()
        time_taken = 'MultinomialNB Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = ngnbC.predict(X_test)
        responce=[]
        responce.append('MultinomialNB Classifier')
        responce.append(accuracy_score(y_test,predictions))
        responce.append(confusion_matrix(y_test,predictions))
        responce.append(pd.DataFrame(classification_report(y_test,predictions, output_dict=True)).transpose())

        from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support=score(y_test,predictions,average='weighted')
        accuracy = accuracy_score(y_test,predictions)
        ac.append(accuracy)
        pr.append(precision)
        re.append(recall)
        f1.append(fscore)
        return (responce)

@app.route("/preclustering", methods=['GET', 'POST'])
def Preclustering():
    if request.method == 'POST':
        # Just render the initial form, to get input
        return render_template('preclustering.html')

@app.route("/handleUploadLabel", methods=['POST'])
def handleFileUploadLabel():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '': 
            path = os.path.join('Upload/', '2.csv')
            photo.save(path)
    return redirect(url_for('Preclustering'))


@app.route('/prediction_clustering', methods=['GET', 'POST'])
def Prediction_clustering():
    if request.method == "POST":
        selected_column = request.form.getlist("column")

    if request.method == "POST":
        df = pd.read_csv('Upload/1.csv', sep=',')
        size=df.size
        dfl = pd.read_csv('Upload/2.csv', sep=',')
        sizel=dfl.size
        pattern = request.form.get("pattern")
        catagories = request.form.get("catagories")
        epsmin = request.form.get("mineps")
        epsmax = request.form.get("maxeps")
        minq = request.form.get("minquantile")
        maxq =request.form.get("maxquantile")

    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('clustering.html')
    select=[]
    print("unlabeled")
    for i in range(9):
        select.append('white')
    if pattern=='no' and catagories=='yes':
        select[6]='green'
        select[4]='aqua'
    if pattern=='no' and catagories=='yes' :
        if size>10000:
            select[0]='green'
            select[1]= 'green'
        else:
            select[1]='green'
            select[6]='aqua'
            select[8]='aqua'
    elif pattern=='no' and catagories=='no' :
        select[4]='green'
        select[5]='aqua'
    elif pattern=='yes' and catagories=='no' : 
        select[3]='green'
        select[2]='aqua'

    matrix=[]
    matrix.append(mbkmean())
    matrix.append(kmean())
    matrix.append(AffPropagation())
    matrix.append(MShift(minq,maxq))
    matrix.append(dbs(epsmin,epsmax))
    matrix.append(opt())
    matrix.append(spectral())
    matrix.append(birch())
    matrix.append(gmm())
    
    if request.method == 'POST':
        # Extract the input
        return render_template('clustering.html', mat=matrix,color=select,i=2)


@app.route('/mbkmean', methods=['GET', 'POST'])
def mbkmean():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'POST':
        km=[]
        sil=[]
        db=[]
        ch=[]
         

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("mbk ",i)
            kmeans = MiniBatchKMeans(n_clusters=i, random_state=0, batch_size=6)
            start_time = time.time()
            labels = kmeans.fit_predict(data)
            end_time = time.time()
            km.append(kmeans.score(data))
            sil.append(silhouette_score(data,labels))
            db.append(davies_bouldin_score(data, labels))
            ch.append(calinski_harabasz_score(data, labels))
            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot1.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(km)
        responce.append(sil)
        responce.append(db)
        responce.append(ch)
        print('kmeans complete')
        return (responce)


@app.route('/kmean', methods=['GET', 'POST'])
def kmean():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        km=[]
        sil=[]
        db=[]
        ch=[]
         

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("k ",i)
            kmeans = KMeans(n_clusters=i)
            start_time = time.time()
            labels = kmeans.fit_predict(data)
            end_time = time.time()
            km.append(kmeans.score(data))
            sil.append(silhouette_score(data,labels))
            db.append(davies_bouldin_score(data, labels))
            ch.append(calinski_harabasz_score(data, labels))
            
            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot2.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(km)
        responce.append(sil)
        responce.append(db)
        responce.append(ch)

        print('kmeans complete')
        
        return (responce)

@app.route('/AffPropagation', methods=['GET', 'POST'])
def AffPropagation():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        sil=[]
        db=[]
        ch=[]
         
        nc=[]
        i = 20

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        while i < 210:
            try:
                print('aff %d' %i)
                af_model= AffinityPropagation(damping=0.9, preference=-i).fit(data)
                cluster_centers_indices = af_model.cluster_centers_indices_
                labels = af_model.labels_
                nc.append(len(cluster_centers_indices))
                if len(labels)>1:
                    sil.append(silhouette_score(data, labels, metric='sqeuclidean'))
                    db.append(davies_bouldin_score(data, labels))
                    ch.append(calinski_harabasz_score(data, labels))
                i += 20

                plt.subplot(1, 10, plot_num)
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#7c5999', '#e41a1c', '#dede00','#600628']),
                                            int(max(labels) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
            except ValueError as err:
                print(err) 

        plt.savefig('static/plot3.png', dpi=300, bbox_inches='tight')


        responce=[]
        responce.append(nc)
        responce.append(sil)
        responce.append(db)
        responce.append(ch)
        
    return (responce)



@app.route('/MShift', methods=['GET', 'POST'])
def MShift(minq,maxq):

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        sil=[]
        db=[]
        ch=[]
        
        nc=[]
        i = minq
        count=(maxq-minq)/10
        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        while i < maxq:
            try:
                bandwidth = estimate_bandwidth(data, quantile=i)
                print("quantile %d", i)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(data)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                n_clusters = len(cluster_centers)
                nc.append(n_clusters)
                if len(labels)>1:
                    sil.append(silhouette_score(data, labels, metric='sqeuclidean'))
                    db.append(davies_bouldin_score(data, labels))
                    ch.append(calinski_harabasz_score(data, labels))
                i +=count

                plt.subplot(1, 10, plot_num)
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#7c5999', '#e41a1c', '#dede00','#600628']),
                                            int(max(labels) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
            except ValueError as err:
                print(err)
                break
            


        plt.savefig('static/plot4.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(nc)
        responce.append(sil)
        responce.append(db)
        responce.append(ch)


    return (responce)


@app.route('/dbs', methods=['GET', 'POST'])
def dbs(epsmin,epsmax):

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        sil=[]
        dbm=[]
        ch=[]
         
        nc=[]
        nn=[]
        #0.1 is min value
        i = epsmin
        count=(epsmax-epsmin)/10
        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1
        
        while i < epsmax:
            try:
                print("eps %d", i)
                db = DBSCAN(eps=i).fit(data)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                # Number of clusters in labels, ignoring noise if present.
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                nc.append(n_clusters)
                nn.append(n_noise)
                if len(labels)>1:
                    sil.append(silhouette_score(data, labels, metric='sqeuclidean'))
                    dbm.append(davies_bouldin_score(data, labels))
                    ch.append(calinski_harabasz_score(data, labels))
                i+=count

                plt.subplot(1, 10, plot_num)
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#7c5999', '#e41a1c', '#dede00','#600628']),
                                            int(max(labels) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
            except ValueError as err:
                print(err)
                break

        plt.savefig('static/plot5.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(nc)
        responce.append(nn)
        responce.append(sil)
        responce.append(dbm)
        responce.append(ch)

    return (responce)



@app.route('/opt', methods=['GET', 'POST'])
def opt():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        sil=[]
        db=[]
        ch=[]
         
        nc=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        i = 0.020
        #1
        while i < 0.4:
            try:
                print("opt",i)
                clust = OPTICS(min_samples=20, xi=i, min_cluster_size=0.1)
                # Run the fit
                clust.fit(data)
                labels = clust.labels_.astype(np.int)
                #space = np.arange(len(data))
                #reachability = clust.reachability_[clust.ordering_]
                labels_unique = np.unique(labels)
                n_clusters = len(labels_unique)
                nc.append(n_clusters)
                sil.append(silhouette_score(data, labels, metric='sqeuclidean'))
                db.append(davies_bouldin_score(data, labels))
                ch.append(calinski_harabasz_score(data, labels))
                i +=0.04

                plt.subplot(1, 10, plot_num)
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#7c5999', '#e41a1c', '#dede00','#600628']),
                                            int(max(labels) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
            except ValueError as err:
                print(err)
                break
        plt.savefig('static/plot6.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(nc)
        responce.append(sil)
        responce.append(db)
        responce.append(ch)

    return (responce)


@app.route('/spectral', methods=['GET', 'POST'])
def spectral():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        sil=[]
        db=[]
        ch=[]
         

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("spec",i)
            start_time = time.time()
            spectral = SpectralClustering(n_clusters=i, eigen_solver='arpack', affinity="nearest_neighbors")
            labels = spectral.fit_predict(data)
            end_time = time.time()
            sil.append(silhouette_score(data,labels))
            db.append(davies_bouldin_score(data, labels))
            ch.append(calinski_harabasz_score(data, labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot7.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(sil)
        responce.append(db)
        responce.append(ch)
        print('spectral complete')
        
        return (responce)

@app.route('/ward', methods=['GET', 'POST'])
def ward():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        ws=[]
        sil=[]
        db=[]
        ch=[]
         

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("ward",i)
            start_time = time.time()
            connectivity = kneighbors_graph(data, n_neighbors=15, include_self=False)
            ward = AgglomerativeClustering(linkage='ward', n_clusters=i, connectivity=connectivity)
            labels = ward.fit_predict(data)
            end_time = time.time()
            ws.append(ward.score(data))
            sil.append(silhouette_score(data,labels))
            db.append(davies_bouldin_score(data, labels))
            ch.append(calinski_harabasz_score(data, labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot8.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(ws)
        responce.append(sil)
        responce.append(db)
        responce.append(ch)
        print('ward complete')
        
        return (responce)


@app.route('/AgglomerativeClustering', methods=['GET', 'POST'])
def AgglomerativeClustering():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        al=[]
        sil=[]
        db=[]
        ch=[]
         

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("agg",i)
            start_time = time.time()
            connectivity = kneighbors_graph(data, n_neighbors=15, include_self=False)
            connectivity = 0.5 * (connectivity + connectivity.T)
            average_linkage = AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=i, connectivity=connectivity)
            labels = average_linkage.fit_predict(data)
            end_time = time.time()
            al.append(average_linkage.score(data))
            sil.append(silhouette_score(data,labels))
            db.append(davies_bouldin_score(data, labels))
            ch.append(calinski_harabasz_score(data, labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot9.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(al)
        responce.append(sil)
        responce.append(db)
        responce.append(ch)
        print('agg complete')
        
        return (responce)

@app.route('/birch', methods=['GET', 'POST'])
def birch():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        bs=[]
        sil=[]
        db=[]
        ch=[]
         

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("birch",i)
            start_time = time.time()
            birch = Birch(n_clusters=i)
            labels = birch.fit_predict(data)
            end_time = time.time()
            sil.append(silhouette_score(data,labels))
            db.append(davies_bouldin_score(data, labels))
            ch.append(calinski_harabasz_score(data, labels))
            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot10.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(sil)
        responce.append(db)
        responce.append(ch)
        print('birch complete')
        
        return (responce)

@app.route('/gmm', methods=['GET', 'POST'])
def gmm():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        gs=[]
        sil=[]
        db=[]
        ch=[]
         

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("gmm",i)
            start_time = time.time()
            gmm = GaussianMixture(n_components=i, covariance_type='full')
            labels = gmm.fit_predict(data)
            end_time = time.time()
            gs.append(gmm.score(data))
            sil.append(silhouette_score(data,labels))
            db.append(davies_bouldin_score(data, labels))
            ch.append(calinski_harabasz_score(data, labels))
            
            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot11.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(gs)
        responce.append(sil)
        responce.append(db)
        responce.append(ch)
        print('gmm complete')
        
        return (responce)


@app.route('/prediction_clusteringLables', methods=['GET', 'POST'])
def Prediction_clusteringLables():
    if request.method == "POST":
        selected_column = request.form.getlist("column")

    if request.method == "POST":
        df = pd.read_csv('Upload/1.csv', sep=',')
        size=df.size
        dfl = pd.read_csv('Upload/2.csv', sep=',')
        sizel=dfl.size
        if size/2 !=sizel :
            error='Size Mismatch'
            print(size," ",sizel)
            return render_template('preclustering.html',error=error)
        pattern = request.form.get("pattern")
        catagories = request.form.get("catagories")
        epsmin = request.form.get("mineps")
        epsmax = request.form.get("maxeps")
        minq = request.form.get("minquantile")
        maxq =request.form.get("maxquantile")

    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('clustering.html')
    select=[]
    for i in range(9):
        select.append('white')
    if pattern=='no' and catagories=='yes':
        select[6]='green'
        select[4]='aqua'
    if pattern=='no' and catagories=='yes' :
        if size>10000:
            select[0]='green'
            select[1]= 'green'
        else:
            select[1]='green'
            select[6]='aqua'
            select[8]='aqua'
    elif pattern=='no' and catagories=='no' :
        select[4]='green'
        select[5]='aqua'
    elif pattern=='yes' and catagories=='no' : 
        select[3]='green'
        select[2]='aqua'
    print("with label")
    matrix=[]
    matrix.append(mbkmeanLabels())
    matrix.append(kmeanLabels())
    matrix.append(AffPropagationLabels())
    matrix.append(MShiftLabels())
    matrix.append(dbsLabels())
    matrix.append(optLabels())
    matrix.append(spectralLabels())
    matrix.append(birchLabels())
    matrix.append(gmmLabels())
    
    if request.method == 'POST':
        # Extract the input
        return render_template('clusteringlabels.html', mat=matrix,color=select,i=2)


@app.route('/mbkmeanLabels', methods=['GET', 'POST'])
def mbkmeanLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'POST':
        km=[]
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("mbk ",i)
            kmeans = MiniBatchKMeans(n_clusters=i, random_state=0, batch_size=6)
            start_time = time.time()
            labels = kmeans.fit_predict(data)
            end_time = time.time()
            km.append(kmeans.score(data))
            ari.append(adjusted_rand_score(dfl,labels))
            mibs.append(adjusted_mutual_info_score(dfl,labels))
            homo.append(homogeneity_score(dfl,labels))
            comp.append(completeness_score(dfl,labels))
            vm.append(v_measure_score(dfl,labels))
            
            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot1.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(km)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)
        print('kmeans complete')
        return (responce)


@app.route('/kmeanLabels', methods=['GET', 'POST'])
def kmeanLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        km=[]
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("k ",i)
            kmeans = KMeans(n_clusters=i)
            start_time = time.time()
            labels = kmeans.fit_predict(data)
            end_time = time.time()
            km.append(kmeans.score(data))
            ari.append(adjusted_rand_score(dfl,labels))
            mibs.append(adjusted_mutual_info_score(dfl,labels))
            homo.append(homogeneity_score(dfl,labels))
            comp.append(completeness_score(dfl,labels))
            vm.append(v_measure_score(dfl,labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot2.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(km)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)
        print('kmeans complete')
        
        return (responce)

@app.route('/AffPropagationLabels', methods=['GET', 'POST'])
def AffPropagationLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]
        nc=[]
        i = 20

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        while i < 200:
            try:
                print('aff %d' %i)
                af_model= AffinityPropagation(damping=0.9, preference=-i).fit(data)
                cluster_centers_indices = af_model.cluster_centers_indices_
                labels = af_model.labels_
                nc.append(len(cluster_centers_indices))
                if len(labels)>1:
                    ari.append(adjusted_rand_score(dfl,labels))
                    mibs.append(adjusted_mutual_info_score(dfl,labels))
                    homo.append(homogeneity_score(dfl,labels))
                    comp.append(completeness_score(dfl,labels))
                    vm.append(v_measure_score(dfl,labels))
                i += 20

                plt.subplot(1, 10, plot_num)
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#7c5999', '#e41a1c', '#dede00','#600628']),
                                            int(max(labels) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
            except ValueError as err:
                print(err) 

        plt.savefig('static/plot3.png', dpi=300, bbox_inches='tight')


        responce=[]
        responce.append(nc)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)
        
    return (responce)



@app.route('/MShiftLabels', methods=['GET', 'POST'])
def MShiftLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]
        nc=[]
        i = 0.02

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        while i < 0.23:
            try:
                bandwidth = estimate_bandwidth(data, quantile=i)
                print("quantile %d", i)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(data)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                n_clusters = len(cluster_centers)
                nc.append(n_clusters)
                if len(labels)>1:
                    ari.append(adjusted_rand_score(dfl,labels))
                    mibs.append(adjusted_mutual_info_score(dfl,labels))
                    homo.append(homogeneity_score(dfl,labels))
                    comp.append(completeness_score(dfl,labels))
                    vm.append(v_measure_score(dfl,labels))                
                i +=0.02

                plt.subplot(1, 10, plot_num)
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#7c5999', '#e41a1c', '#dede00','#600628']),
                                            int(max(labels) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
            except ValueError as err:
                print(err)
                break
            


        plt.savefig('static/plot4.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(nc)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)

    return (responce)


@app.route('/dbsLabels', methods=['GET', 'POST'])
def dbsLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]
        nc=[]
        nn=[]
        #0.1 is min value
        i = 0.1

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1
        
        while i < 0.3:
            try:
                print("eps %d", i)
                db = DBSCAN(eps=i).fit(data)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                # Number of clusters in labels, ignoring noise if present.
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                nc.append(n_clusters)
                nn.append(n_noise)
                if len(labels)>1:
                    ari.append(adjusted_rand_score(dfl,labels))
                    mibs.append(adjusted_mutual_info_score(dfl,labels))
                    homo.append(homogeneity_score(dfl,labels))
                    comp.append(completeness_score(dfl,labels))
                    vm.append(v_measure_score(dfl,labels))
                i+=0.02

                plt.subplot(1, 10, plot_num)
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#7c5999', '#e41a1c', '#dede00','#600628']),
                                            int(max(labels) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
            except ValueError as err:
                print(err)
                break

        plt.savefig('static/plot5.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(nc)
        responce.append(nn)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)

    return (responce)



@app.route('/optLabels', methods=['GET', 'POST'])
def optLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        nc=[]
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        i = 0.025
        #1
        while i < 0.26:
            try:
                print("opt",i)
                clust = OPTICS(min_samples=20, xi=i, min_cluster_size=0.1)
                # Run the fit
                clust.fit(data)
                labels = clust.labels_.astype(np.int)
                space = np.arange(len(data))
                reachability = clust.reachability_[clust.ordering_]
                labels = clust.labels_[clust.ordering_]
                labels_unique = np.unique(labels)
                n_clusters = len(labels_unique)
                nc.append(n_clusters)
                ari.append(adjusted_rand_score(dfl,labels))
                mibs.append(adjusted_mutual_info_score(dfl,labels))
                homo.append(homogeneity_score(dfl,labels))
                comp.append(completeness_score(dfl,labels))
                vm.append(v_measure_score(dfl,labels))
                i +=0.025

                plt.subplot(1, 10, plot_num)
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#7c5999', '#e41a1c', '#dede00','#600628']),
                                            int(max(labels) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
            except ValueError as err:
                print(err)
                break
        plt.savefig('static/plot6.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(nc)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)

    return (responce)


@app.route('/spectralLabels', methods=['GET', 'POST'])
def spectralLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("spec",i)
            start_time = time.time()
            spectral = SpectralClustering(n_clusters=i, eigen_solver='arpack', affinity="nearest_neighbors")
            labels = spectral.fit_predict(data)
            end_time = time.time()
            ari.append(adjusted_rand_score(dfl,labels))
            mibs.append(adjusted_mutual_info_score(dfl,labels))
            homo.append(homogeneity_score(dfl,labels))
            comp.append(completeness_score(dfl,labels))
            vm.append(v_measure_score(dfl,labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot7.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)
        print('spectral complete')
        
        return (responce)

@app.route('/wardLabels', methods=['GET', 'POST'])
def wardLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        al=[]
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("ward",i)
            start_time = time.time()
            connectivity = kneighbors_graph(data, n_neighbors=i, include_self=False)
            ward = AgglomerativeClustering(linkage='ward', n_clusters=i, connectivity=connectivity)
            labels = ward.fit_predict(data)
            end_time = time.time()
            ws.append(ward.score(data))
            ari.append(adjusted_rand_score(dfl,labels))
            mibs.append(adjusted_mutual_info_score(dfl,labels))
            homo.append(homogeneity_score(dfl,labels))
            comp.append(completeness_score(dfl,labels))
            vm.append(v_measure_score(dfl,labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot8.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(ws)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)
        print('agg complete')
        
        return (responce)


@app.route('/AgglomerativeClusteringLabels', methods=['GET', 'POST'])
def AgglomerativeClusteringLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        al=[]
        ari=[]
        mibs=[]
        ch=[]
        comp=[]
        vm=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("agg",i)
            start_time = time.time()
            connectivity = kneighbors_graph(data, n_neighbors=i, include_self=False)
            connectivity = 0.5 * (connectivity + connectivity.T)
            average_linkage = AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=i, connectivity=connectivity)
            labels = spectral.fit_predict(data)
            end_time = time.time()
            al.append(average_linkage.score(data))
            ari.append(adjusted_rand_score(dfl,labels))
            mibs.append(adjusted_mutual_info_score(dfl,labels))
            ch.append(homogeneity_score(dfl,labels))
            comp.append(completeness_score(dfl,labels))
            vm.append(v_measure_score(dfl,labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot9.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(al)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)
        print('agg complete')
        
        return (responce)

@app.route('/birchLabels', methods=['GET', 'POST'])
def birchLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        bs=[]
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("birch",i)
            start_time = time.time()
            birch = Birch(n_clusters=i)
            labels = birch.fit_predict(data)
            end_time = time.time()
            ari.append(adjusted_rand_score(dfl,labels))
            mibs.append(adjusted_mutual_info_score(dfl,labels))
            homo.append(homogeneity_score(dfl,labels))
            comp.append(completeness_score(dfl,labels))
            vm.append(v_measure_score(dfl,labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot10.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)
        print('birch complete')
        
        return (responce)

@app.route('/gmmLabels', methods=['GET', 'POST'])
def gmmLabels():

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    dflabel = pd.read_csv('Upload/2.csv', sep=',').to_numpy()
    dfl = dflabel.flatten()
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('clustering.html'))
    
    if request.method == 'POST':
        gs=[]
        ari=[]
        mibs=[]
        homo=[]
        comp=[]
        vm=[]

        plt.figure(figsize=(9 * 2 + 3, 2.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plot_num = 1

        for i in range(2,12):
            print("gmm",i)
            start_time = time.time()
            gmm = GaussianMixture(n_components=i, covariance_type='full')
            labels = gmm.fit_predict(data)
            end_time = time.time()
            gs.append(gmm.score(data))
            ari.append(adjusted_rand_score(dfl,labels))
            mibs.append(adjusted_mutual_info_score(dfl,labels))
            homo.append(homogeneity_score(dfl,labels))
            comp.append(completeness_score(dfl,labels))
            vm.append(v_measure_score(dfl,labels))

            plt.subplot(1, 10, plot_num)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#7c5999', '#e41a1c', '#dede00','#600628']),
                                        int(max(labels) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(data[:,0], data[:,1], s=10, color=colors[labels])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

        plt.savefig('static/plot11.png', dpi=300, bbox_inches='tight')

        responce=[]
        responce.append(gs)
        responce.append(ari)
        responce.append(mibs)
        responce.append(homo)
        responce.append(comp)
        responce.append(vm)
        print('gmm complete')
        
        return (responce)




if __name__ == '__main__':
    app.run(debug=True)

     
