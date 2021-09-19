import torch, pickle
from sklearn.ensemble import RandomForestClassifier


class RandomForestPartition:
    def __init__(self, model, n_estimators=10, random_state=0):
        self.model = model
        self.n_estimators = n_estimators
        self.random_state = random_state

    @classmethod
    def from_train(cls, D, n_estimators=10, random_state=0):
        x_train, y_train = D
        x_train = x_train.squeeze(2)
        model = RandomForestClassifier(n_estimators, random_state=random_state)
        model.fit(x_train, y_train)
        return cls(model, n_estimators, random_state)

    @classmethod
    def load(cls, filename, n_estimators=10, random_state=0):
        # Load from file
        with open(filename, "rb") as file:
            model = pickle.load(file)
        return cls(model, n_estimators, random_state)

    def predict(self, X):
        X = X.squeeze(2)
        if X.is_cuda:
            return torch.from_numpy(self.model.predict(X.cpu())).to("cuda:0")
        return torch.from_numpy(self.model.predict(X))

    def score(self, X, Y):
        X = X.squeeze(2)
        if X.is_cuda and Y.is_cuda:
            return self.model.score(X.cpu(), Y.cpu())
        return self.model.score(X, Y)

    def to_string(self):
        return "RandomForest_" + str(self.n_estimators)

    def store(self, filename):
        # Save to file in the current working directory
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
