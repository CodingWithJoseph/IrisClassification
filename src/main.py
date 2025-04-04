import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import fetch_and_format_iris, test_train_split
from model import SoftmaxRegression
from sklearn.metrics import classification_report

if __name__ == '__main__':
    data = fetch_and_format_iris()
    X, X_test, y, y_test = test_train_split(data)
    model = SoftmaxRegression()

    epochs = 1000
    learning_rate = 1e-2
    model.train(X, y, learning_rate, epochs)

    model_predictions_train = model.predict(X)
    model_predictions_test = model.predict(X_test)

    print('Training accuracy', np.round(np.mean(model_predictions_train == y), 3))
    print('Test accuracy', np.round(np.mean(model_predictions_test == y_test), 3))
    print(classification_report(y_test, model_predictions_test, digits=3))

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
