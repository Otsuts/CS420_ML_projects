from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class classifier():
    def __init__(self, args) -> None:
        if args.model == 'mlp_big':
            self.clf = MLPClassifier(
                solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        elif args.model == 'svm':
            self.clf = SVC(C=args.C, kernel=args.kernel,
                           decision_function_shape='ovr')
        elif args.model == 'mlp_small':
            self.clf = MLPClassifier(
                solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), max_iter=1000, random_state=42)


    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
