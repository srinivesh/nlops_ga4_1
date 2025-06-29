# experiment with pytest

import unittest
import pickle
import numpy as np
import sys
import os

import pandas as pd
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class TestModel(unittest.TestCase):
    restored_model = None
    base_model = None
    top_model_complete = 'model.keras'

    def setUp(self):

        self.restored_model = pickle.load(open('knnpickle_file', 'rb'))


    def load_data(self):
        data = 'sample_data.csv.test'
        column_headers = ['sepal-length', 'sepal-width',
                  'petal-length', 'petal-width', 'class']
        df = pd.read_csv(data, names=column_headers)

        data = df.values
        return data

    def test_sample1(self):
        
        data = self.load_data()
        X = data[:, 0:4]
        y = data[:, 4]

        yhat = self.restored_model.predict(X)
        accuracy = accuracy_score(y, yhat)

        #use accuracy of 0.97 as pass criteria
        assert accuracy >= 0.97
        
        #create some output 
        with open('test_output.txt', 'w') as f:
            msg = "Model Accuracy: %.2f" %(accuracy)
            f.write(msg)
            f.write("\n")
            print(msg)
       # self.assertEqual(prediction,"dog","Predicted class is wrong")


if __name__ == '__main__':
    unittest.main()
