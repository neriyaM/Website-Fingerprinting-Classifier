import os.path
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import csv
from sklearn.utils import shuffle
import numpy as np


class DataLoader:
    def __init__(self, embedding_model, path):
        self.embedding_model = embedding_model
        self.train_path = os.path.join(path, 'train')
        self.test_path = os.path.join(path, 'test')
        classes = get_classes_name(self.train_path)
        le = preprocessing.LabelEncoder()
        le.fit(classes)
        self.le = le

    def load_data(self):
        X_train, Y_train = self._load_data(self.train_path)
        X_test, Y_test = self._load_data(self.test_path)
        return X_train, Y_train, X_test, Y_test

    def _load_data(self, path):
        X, Y = [], []
        filenames = get_file_names(path)
        for filename in filenames:
            label = self.le.transform([filename.split(".")[0]])[0]
            file_path = os.path.join(path, filename)
            features = read_csv_file(file_path)
            for i in features:
                vector = [int(float(j)) for j in i]
                embedding_vector = self.embedding_model.predict(np.array(vector).reshape(1, 1000))
                X.append(embedding_vector.reshape(128))
                Y.append(label)

        return shuffle(X, Y)


def get_classes_name(path):
    return [f.split(".")[0] for f in listdir(path) if isfile(join(path, f))]


def read_csv_file(file_path):
    file = open(file_path)
    reader = csv.reader(file)
    return list(reader)


def get_file_names(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
