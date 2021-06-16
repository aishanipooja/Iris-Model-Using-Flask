from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('iri.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def iris():
    d1 = request.form['a']
    d2 = request.form['b']
    d3 = request.form['c']
    d4 = request.form['d']
    arr = np.array([[d1, d2, d3, d4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == '__main__':
    app.run()
