import flask
from flask import Flask, request, render_template, url_for, redirect
import pandas as pd
import numpy as np
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





data= pd.read_csv("winequality-red.csv")

#Feature Extractions and Partitions
X=pd.DataFrame(data.iloc[:,:-1])
y=pd.DataFrame(data.iloc[:,-1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)




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
                                     

@app.route('/decisiontreeC', methods=['GET', 'POST'])
def decisiontreeC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        #decision tree classifier
        from sklearn.tree import DecisionTreeClassifier
        decision_tree = DecisionTreeClassifier()
        decision_tree = decision_tree.fit(X_train,y_train)
        
        #from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

        predictions = decision_tree.predict(X_test)
    
        acc=accuracy_score(predictions, y_test)
# But Confusion Matrix and Classification Report give more details about performance
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test)
        a=np.array(cf)
        b=np.array(cr)
        
        print(a)
        print(b)

        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=a,
                                     result2=b,
                                     )          

@app.route('/randomforestC', methods=['GET', 'POST'])
def randomforestC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        random_forest = RandomForestClassifier()
        random_forest.fit(X_train,y_train)
        
        predictions = random_forest.predict(X_test)

        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test)        
        
        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=cr,
                                     )   

@app.route('/logisticR', methods=['GET', 'POST'])
def logisticR():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        logistic = LogisticRegression()
        logistic.fit(X_train,y_train)
        predictions = logistic.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test)        

        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=cr,
                                     )  

@app.route('/svmC', methods=['GET', 'POST'])
def svmC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        support_vector = SVC()
        support_vector.fit(X_train,y_train)
        predictions = support_vector.predict(X_test)        
        
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test)        

        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=cr,
                                     ) 

@app.route('/knnC', methods=['GET', 'POST'])
def knnC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        KNN = KNeighborsClassifier(n_neighbors=5)
        KNN.fit(X_train,y_train)
        predictions = KNN.predict(X_test)
        acc=accuracy_score(predictions, y_test)
        cf=confusion_matrix(predictions, y_test)
        cr=classification_report(predictions, y_test)        

        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=cr,
                                     )


if __name__ == '__main__':
    app.run(debug=True)