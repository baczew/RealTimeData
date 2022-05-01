


from sklearn import datasets
import numpy as np


iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


#%%file app_h.py

from sklearn import datasets
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import numpy as np

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


mod = Perceptron()
mod.fit(X, y)

app = Flask(__name__)

api = Api(app)

@app.route('/api/predict/')
def home():
    sl = request.args.get('sl', 0)
    pl = request.args.get('pl', 0)
    eX = np.array([float(sl), float(pl)])
    return '<h1>Strona Początkowa</h1>' + '\n<h2>' + str(sl) + '</h2>\n<h2>' + str(pl) \
            + '</h2>' + str(mod.predict(eX)) 



app.run(port=5596)


import subprocess
import requests
p = subprocess.Popen(['python3', 'app_h.py'])


import requests
resp = requests.get('http://127.0.0.1:5596/api/predict/?&sl=4.5&pl=3.2')
print(resp)#4.5 3.2


resp.text

