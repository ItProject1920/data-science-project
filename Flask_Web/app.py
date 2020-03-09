#open anaconda base environment and app.py in it
import os
from flask import Flask, flash, request, render_template, url_for, redirect
#from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from werkzeug.utils import secure_filename


app = Flask(__name__)
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
            path = os.path.join('/home/aditya/Documents/Uploads/', '1.csv')
            photo.save(path)
    return redirect(url_for('fileFrontPage'))

@app.route("/suggest")
def suggest():
    return render_template('display.html')
	
@app.route("/display")
def Display():
    df = pd.read_csv('/home/aditya/Documents/Uploads/1.csv', sep=',')
    sf = df.head(5)
    i = list(df.columns.values)
    return render_template('display.html', tables=[sf.to_html()], index=i)

@app.route('/history')
def history():
    return render_template('history.html') 


if __name__ == '__main__':
    app.run(debug=True)     