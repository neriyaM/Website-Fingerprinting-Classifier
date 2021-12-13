from keras.models import load_model
from keras.models import Model
import loader
from classifier import Classifier


def get_embedding_model(path, layer_name):
    model = load_model(path)
    base_layer = model.get_layer(layer_name)
    return Model(inputs=base_layer.input, outputs=base_layer.output)


def main():
    embedding_model = get_embedding_model('model.h5', 'model')
    data_loader = loader.DataLoader(embedding_model, 'minidata')
    X_train, Y_train, X_test, Y_test = data_loader.load_data()

    classifier = Classifier(3)
    classifier.train(X_train, Y_train)
    confusion_matrix, report, accuracy = classifier.evaluate(X_test, Y_test)
    print(confusion_matrix)
    print(report)
    print(accuracy)
    print("WOWWWWW")


main()

# x = embedding_model.predict(np.zeros(1000).reshape(1, 1000))
