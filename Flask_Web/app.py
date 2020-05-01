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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn import neighbors
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
from sklearn.preprocessing import StandardScaler
from flask_jsglue import JSGlue


app = Flask(__name__, template_folder='templates')
#app.config['SQLALCHEMY_DATABASE_URL'] = 'sqlite:///test.db'
#rom flask_sqlalchemy import SQLAlchemydb = SQLAlchemy(app)

jsglue = JSGlue(app)

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
	
@app.route("/display")
def Display():
    df = pd.read_csv('Upload/1.csv', sep=',')
    sf = df.head(5)
    i = list(df.columns.values)
    return render_template('display_cal.html', tables=[sf.to_html()], index=i)

@app.route("/display_pre")
def Display_pre():
    df = pd.read_csv('Upload/1.csv', sep=',')
    sf = df.head(5)
    i = list(df.columns.values)
    return render_template('display_pre.html', tables=[sf.to_html()], index=i)

@app.route('/history')
def history():
    return render_template('history.html') 


@app.route('/comparison', methods=['GET', 'POST'])
def comparison():
    algorithms=['xgboost','linearR','decisiontreeR', 'ridgeR', 'lassoR', 'knnR']
    
    
    if request.method == "POST":
        selected_column = request.form.getlist("column")

    if request.method == "POST":
        selected_predict = request.form.getlist("predict")

    df = pd.read_csv('Upload/1.csv', sep=',')


    x = df.loc[:,selected_column]
    y = df.loc[:,selected_predict]

    
    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    
    matrix=[]
    matrix.append(xgboost())
    matrix.append(linearR())
    matrix.append(decisiontreeR())
    matrix.append(ridgeR())
    matrix.append(lassoR())
    matrix.append(knnR())

    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        # Extract the input
        return render_template('regression.html',sc=selected_column, algo= algorithms, mat=matrix)

                                    
@app.route('/xgboost', methods=['GET', 'POST'])
def xgboost():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        # Extract the input
        
        classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
        classifier.fit(X_train, y_train)

        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = classifier.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
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
        return (responce)

@app.route('/decisiontreeR', methods=['GET', 'POST'])
def decisiontreeR():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('regression.html')
    
    if request.method == 'POST':
        # Extract the input
        classifier=DecisionTreeRegressor(max_depth=5,random_state=0)

        classifier.fit(X_train, y_train)

        
        from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

        predictions = classifier.predict(X_test)
        responce=[]
        responce.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        responce.append(mean_squared_error(y_true=y_test, y_pred=predictions))
        responce.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
        responce.append(explained_variance_score(y_true=y_test, y_pred=predictions))
        responce.append(r2_score(y_true=y_test, y_pred=predictions))
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
        return (responce)                                                            

@app.route('/prediction_classification', methods=['GET', 'POST'])
def Prediction_classification():
    if request.method == "POST":
        selected_column = request.form.getlist("column")

    if request.method == "POST":
        selected_predict = request.form.getlist("predict")

    df = pd.read_csv('Upload/1.csv', sep=',')

    x = df.loc[:,selected_column]
    y = df.loc[:,selected_predict]

    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('classification.html')
    
    if request.method == 'POST':
        # Extract the input
        return render_template('classification.html',sc=selected_column)

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
    
        acc=accuracy_score(predictions, y_test)
# But Confusion Matrix and Classification Report give more details about performance
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)
        df = pd.DataFrame(cr).transpose()

        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )          

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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = random_forest.predict(X_test)

        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)        
        df = pd.DataFrame(cr).transpose()
        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )   

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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)
        predictions = logistic.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)        
        df = pd.DataFrame(cr).transpose()
        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )  

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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = support_vector.predict(X_test)        
        
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)        
        df = pd.DataFrame(cr).transpose()
        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     ) 

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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = KNN.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)       
        df = pd.DataFrame(cr).transpose()
        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )

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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = gpC.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)       
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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = mlpC.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)       
        df = pd.DataFrame(cr).transpose()       

        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )
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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = adC.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)       
        df = pd.DataFrame(cr).transpose()       

        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )

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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = nbC.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)       
        df = pd.DataFrame(cr).transpose()        

        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )
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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = qdaC.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)       
        df = pd.DataFrame(cr).transpose()        

        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )

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
        time_taken = 'Decision Tree Classifier took {:.5f} s'.format(end_time - start_time)        
        predictions = ngnbC.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test, output_dict=True)       
        df = pd.DataFrame(cr).transpose()        

        return render_template('classification.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     time = time_taken,
                                     )

@app.route('/prediction_clustering', methods=['GET', 'POST'])
def Prediction_clustering():

   
    if request.method == "POST":
        selected_column = request.form.getlist("column")

    if request.method == "POST":
        selected_predict = request.form.getlist("predict")

    df = pd.read_csv('Upload/1.csv', sep=',')
    data = StandardScaler().fit_transform(df)
    
    x = df.loc[:,selected_column]
    y = df.loc[:,selected_predict]

    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('classification.html')
    
    if request.method == 'POST':
        # Extract the input
        return render_template('classification.html',sc=selected_column)

if __name__ == '__main__':
    app.run(debug=True)

     
