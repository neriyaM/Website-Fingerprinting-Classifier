from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class Classifier:
    def __init__(self, n_neighbors):
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                               weights='distance',
                                               p=1, metric='manhattan',
                                               algorithm='brute')

    def train(self, X, Y):
        self.classifier.fit(X, Y)

    def evaluate(self, X_test, Y_test):
        predicts = self.classifier.predict(X_test)
        return confusion_matrix(Y_test, predicts), classification_report(Y_test, predicts), accuracy_score(Y_test,
                                                                                                           predicts)
