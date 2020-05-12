import io
import random
from flask import Flask, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from matplotlib.figure import Figure




app = Flask(__name__)


df = pd.read_csv('measures.csv', sep=',')
sf=df.head(5)
print(sf)


X = df['Height (cm)'].values.reshape(-1, 1)
y = df['Weight (kg)'].values.reshape(-1, 1)



@app.route("/")
def index():
    """ Returns html with the img tag for your plot.
    """
    #use img src="the function to be called" in html page to call the function<------this
    # in a real app you probably want to use a flask render_template.
    return f"""
    <h1>Flask and matplotlib</h1>

    <h3>Plot as a png</h3>
    <img src="/matplot-as-image.png"
         alt="random points as png"
         height="400"
    >

    """
    # from flask import render_template
    # return render_template("yourtemplate.html", num_x_points=num_x_points)


@app.route("/matplot-as-image.png")
def plot_png():
    """ renders the plot on the fly.
    """
    
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.plot(X,y, 'b*')  #b* means blue color and symbol to be used is *
    #axis.legend(loc='upper center', numpoints=1, bbox_to_anchor=(0.5, -0.05),        ncol=2, fancybox=True, shadow=True)
    #axis.xlabel("n iteration")
    axis.set_xlabel("HEIGHT")
    axis.set_ylabel("WEIGHT") 


    #these lines below
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")




if __name__ == "__main__":
    import webbrowser

    
    app.run(debug=True)
	
	
	
	
	
	
