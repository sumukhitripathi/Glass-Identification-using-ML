import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


class GlassClassifier:

    def __init__(self, filepath):

        print("Loading glass dataset...")

        self.df = pd.read_csv(filepath)


    def prepare_data(self):

        X = self.df.drop(columns=["Type"])
        y = self.df["Type"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        return X_train, X_test, y_train, y_test


    def train_knn(self, X_train, y_train):

        print("Training KNN classifier...")

        model = KNeighborsClassifier(n_neighbors=5)

        model.fit(X_train, y_train)

        return model


    def train_naive_bayes(self, X_train, y_train):

        print("Training Naive Bayes classifier...")

        model = GaussianNB()

        model.fit(X_train, y_train)

        return model


    def train_random_forest(self, X_train, y_train):

        print("Training Random Forest classifier...")

        model = RandomForestClassifier(n_estimators=100)

        model.fit(X_train, y_train)

        return model


    def evaluate(self, model, X_test, y_test):

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions) * 100

        return accuracy


def run_models(filepath):

    classifier = GlassClassifier(filepath)

    X_train, X_test, y_train, y_test = classifier.prepare_data()

    knn = classifier.train_knn(X_train, y_train)
    nb = classifier.train_naive_bayes(X_train, y_train)
    rf = classifier.train_random_forest(X_train, y_train)

    knn_acc = classifier.evaluate(knn, X_test, y_test)
    nb_acc = classifier.evaluate(nb, X_test, y_test)
    rf_acc = classifier.evaluate(rf, X_test, y_test)

    return {
        "KNN": knn_acc,
        "NaiveBayes": nb_acc,
        "RandomForest": rf_acc
    }


if __name__ == "__main__":

    results = run_models("glass.csv")

    print("\nModel Accuracy Comparison:")

    for model, acc in results.items():
        print(f"{model}: {acc:.2f}%")

    best_model = max(results, key=results.get)

    print(f"\nBest Performing Model: {best_model}")