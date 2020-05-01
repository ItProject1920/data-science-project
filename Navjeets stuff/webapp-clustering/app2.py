import flask
from flask import Flask, request, render_template, url_for, redirect
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
#from sklearn.cluster import DBSCAN
#from sklearn.cluster import DBSCAN
#from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score


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
                                     

@app.route('/kmeansC', methods=['GET', 'POST'])
def kmeansC():
    if flask.request.method == 'GET':
  
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        #CODE

        
        #METRICS
        km_scores= []
        km_silhouette = []
        db_score = []
        for i in range(2,12):
            km = KMeans(n_clusters=i,random_state=0)
            km.fit(X_train,y_train)
            preds=km.predict(X_test)
            
            #print("Score for number of cluster(s) {}: {}".format(i,km.score(X_test)))
            km_scores.append(-km.score(X_test))

            
            silhouette = silhouette_score(X_test,preds)
            km_silhouette.append(silhouette)
            #print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
            
            db = davies_bouldin_score(X_test,preds)
            db_score.append(db)
            #print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
            


    return flask.render_template('main2.html', result=km_silhouette,
                                result2=db_score,
                                )          




if __name__ == '__main__':
    app.run(debug=True)