from flask import Flask, render_template, request, url_for
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

app = Flask(__name__)

@app.route('/home')
def home():
    iris = load_iris()
    X_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train,y_train)
    pickle.dump(model,open('iris.pkl','wb'))
    return render_template("index.html")

@app.route("/predict",methods = ["POST","GET"])
def predict():
    if request.method == "POST":
        sepal_length = float(request.form["sepallength"])
        sepal_width = float(request.form["sepalwidth"])
        petal_length = float(request.form["petallength"])
        petal_width = float(request.form["petalwidth"])
        new_array = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
        model = pickle.load(open('iris.pkl',"rb"))
        prediction = model.predict(new_array)[0]
        if prediction == 0:
            result = "Setosa"
            image =  "setosa.jpg"
        elif prediction == 1:
            result = "Virginica"
            image = "virginica.jpg"
        else:
            result = "Versicolor"
            image = "versicolor.jpg"

        return render_template("result.html",result = result , image = image)








if __name__ == "__main__":
    app.run(debug=True)