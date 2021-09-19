import torch, pickle
from sklearn.svm import SVC


class SVMPartition:
    def __init__(self, model, kernel):
        self.model = model
        self.kernel = kernel

    @classmethod
    def from_train(cls, D, kernel="linear"):
        x_train, y_train = D
        x_train = x_train.squeeze(2)
        model = SVC(kernel=kernel)
        model.fit(x_train, y_train)
        return cls(model, kernel=kernel)

    @classmethod
    def load(cls, filename, kernel):
        # Load from file
        with open(filename, "rb") as file:
            model = pickle.load(file)
        return cls(model=model, kernel=kernel)

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
        return "SVM_" + self.kernel

    def store(self, filename):
        # Save to file in the current working directory
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
