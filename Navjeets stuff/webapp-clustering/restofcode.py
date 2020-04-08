@app.route('/apC', methods=['GET', 'POST'])
def apC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        
        ap=AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False)
        ap.fit(X_train,y_train)
        predictions=ap.predict(X_test)

        #METRICS


        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     )  

@app.route('/msC', methods=['GET', 'POST'])
def msC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        ms=MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300)
        ms.fit(X_train,y_train)
        predictions=ms.predict(X_test)

        #METRICS


        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     ) 

@app.route('/sC', methods=['GET', 'POST'])
def knnC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        s=SpectralClustering(n_clusters=8, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None)
        s.fit(X_train,y_train)
        predictions=s.predict(X_test)

        #METRICS


        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     )

@app.route('/dbscanC', methods=['GET', 'POST'])
def dbscanC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        dbscan=DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)          
        dbscan.fit(X_train,y_train)
        predictions=dbscan.predict(X_test)

        #METRICS



        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     )

@app.route('/agC', methods=['GET', 'POST'])
def agC():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))
    
    if flask.request.method == 'POST':
        agC=AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)       
        agC.fit(X_train,y_train)
        predictions=agC.predict(X_test)

        #METRICS

        return flask.render_template('main2.html',
                                     result=acc,
                                     result1=cf,
                                     result2=df,
                                     )



























