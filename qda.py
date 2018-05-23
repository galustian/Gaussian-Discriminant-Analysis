import numpy as np

class QuadraticDiscriminantAnalysis:

    def fit(self, X, y):
        self.labels, self.class_priors = np.unique(y, return_counts=True)
        self.class_priors = self.class_priors / y.shape[0]

        self.Cov = []
        self.Mu = []
        
        for k in range(len(self.labels)):
            X_k = X[y==self.labels[k]]
            self.Mu.append(np.mean(X_k, axis=0))
            self.Cov.append(np.cov(X_k.T))
        
    def predict(self, X):
        labels = []

        for i in range(X.shape[0]):
            labels.append(self.predict_sample(X[i]))
        
        return np.array(labels)

    def predict_sample(self, X):
        max_label = 0
        max_likelihood = 0

        for k in range(len(self.labels)):
            likelihood  = np.exp(-1/2 * (X - self.Mu[k]).T @ np.linalg.inv(self.Cov[k]) @ (X - self.Mu[k]))
            
            if likelihood > max_likelihood:
                max_label = self.labels[k]
                max_likelihood = likelihood
        
        return max_label