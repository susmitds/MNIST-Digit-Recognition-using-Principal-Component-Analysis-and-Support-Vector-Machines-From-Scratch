import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # Subtract the mean from the data
        X = X - np.mean(X, axis=0)
        # Compute the covariance matrix
        cov = np.cov(X.T)
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # Sort the eigenvectors in descending order based on their corresponding eigenvalues
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        # Store the first n_components eigenvectors as components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X) -> np.ndarray:
        # Subtract the mean from the data
        X = X - np.mean(X, axis=0)
        # Transform the data using the components
        transformed_data = np.dot(X, self.components)
        return transformed_data

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        n_features = X.shape[1]
        self.b = 0
        self.w = np.zeros(n_features)
        self.w = np.hstack([[self.b], self.w]) #adding b to weight matrix as w_0

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        
        self._initialize(X)
        X=np.hstack([np.ones([X.shape[0], 1], dtype=np.int32), X]) #adding constant column as X_0 to X
        for i in tqdm(range(1, num_iters + 1), leave=False):
        # fit the SVM model using stochastic gradient descent
            # sample a random training example
            random_id = np.random.choice(X.shape[0], size=1)[0]
            x_i = X[random_id]
            y_i = y[random_id]

            # calculate the hinge loss and its derivative
            Jw = (self.w**2)/2 + np.maximum(0, 1 - y_i * (np.dot(x_i, self.w)))
            delta_Jw = np.zeros_like(self.w)
            if np.maximum(0, 1 - y_i * (np.dot(x_i, self.w)))> 0:
                delta_Jw = self.w - C * y_i * x_i

            # update the parameters using stochastic gradient descent
            self.w -= learning_rate * delta_Jw
            self.b = self.w[0]

    def predict_score(self, X) -> np.ndarray:
        # make predictions for the given data
        X=np.hstack([np.ones([X.shape[0], 1], dtype=np.int32), X])
        return np.dot(X, self.w)

    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return np.sign(self.predict_score(X)).astype(np.int8)

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)

    def precision_score(self, X, y) -> float:
        # compute the precision of the model (for debugging purposes)
        y_pred = self.predict(X)
        tp = 0
        for yt, yp in zip(y, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
        fp = 0
        for yt, yp in zip(y, y_pred):
            if yt == -1 and yp == 1:
                fp += 1
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)

    def recall_score(self, X, y) -> float:
        # compute the recall of the model (for debugging purposes)
        y_pred = self.predict(X)
        tp = 0
        for yt, yp in zip(y, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
        fn = 0
        for yt, yp in zip(y, y_pred):
            if yt == 1 and yp == -1:
                fn += 1
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)


    def f1_score(self, X, y) -> float:
        # compute the F1 score of the model (for debugging purposes)
        precision = self.precision_score(X, y)
        recall = self.recall_score(X, y)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)
        
    def coeff(self):
        return (self.w)

class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def _preprocess(self, y):
        # preprocess the data to make it suitable for the 1-vs-rest SVM model
        y_arr = []
        for i in range(self.num_classes):
            y_new = []
            y_new.append(np.where(y == i, 1, -1))
            y_new = np.hstack(y_new)
            y_arr.append(y_new)
        return y_arr

    def fit(self, X, y, learning_rate: float, num_iters: int, C: float = 1.0) -> None:
        # preprocess the data to make it suitable for the 1-vs-rest SVM model
        y_new = self._preprocess(y)

        # train the 10 SVM models using the preprocessed data for each class
        for i in tqdm(range(self.num_classes)):
            self.models[i].fit(X, y_new[i], learning_rate=learning_rate, num_iters=num_iters, C=C)

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        scores = []
        for i in range(self.num_classes):
            scores.append(self.models[i].predict_score(X))
        scores = np.vstack(scores).T
        return np.argmax(scores, axis=1)
        
    def true_positive(self, y, y_pred):
        tp = 0
        for yt, yp in zip(y, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
        return tp

    def true_negative(self, y, y_pred):
        tn = 0
        for yt, yp in zip(y, y_pred):
            if yt == 0 and yp == 0:
                tn += 1
        return tn

    def false_positive(self, y, y_pred):
        fp = 0
        for yt, yp in zip(y, y_pred):
            if yt == 0 and yp == 1:
                fp += 1
        return fp

    def false_negative(self, y, y_pred):
        fn = 0
        for yt, yp in zip(y, y_pred):
            if yt == 1 and yp == 0:
                fn += 1
        return fn

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model
        return np.mean(self.predict(X) == y)

    def precision_score(self, X, y) -> float:
        # compute the precision of the model
        y_pred = self.predict(X)
        # initialize precision to 0
        precision = 0
        for class_ in list(set(y)):
            actual = [1 if p == class_ else 0 for p in y]
            pred = [1 if p == class_ else 0 for p in y_pred]
            tp = self.true_positive(actual, pred)
            fp = self.false_positive(actual, pred)
            class_precision = tp / (tp + fp + 1e-6) #adding 1e-6 to denominator to avoid DivisionByZero Error
            # keep adding precision for all classes
            precision += class_precision
        # calculate and return average precision over all classes
        precision /= self.num_classes
        return precision

    def recall_score(self, X, y) -> float:
        # compute the recall of the model
        y_pred = self.predict(X)
        # initialize recall to 0
        recall = 0
        for class_ in list(set(y)):
            actual = [1 if p == class_ else 0 for p in y]
            pred = [1 if p == class_ else 0 for p in y_pred]
            tp = self.true_positive(actual, pred)
            fn = self.false_negative(actual, pred)
            class_recall = tp / (tp + fn + 1e-6) #adding 1e-6 to denominator to avoid DivisionByZero Error
            # keep adding recall for all classes
            recall += class_recall
        # calculate and return average recall over all classes
        recall /= self.num_classes
        return recall

    def f1_score(self, X, y) -> float:
        # compute the F1 score of the model
        y_pred = self.predict(X)
        # initialize f1 to 0
        f1 = 0
        for class_ in list(set(y)):
            actual = [1 if p == class_ else 0 for p in y]
            pred = [1 if p == class_ else 0 for p in y_pred]
            tp = self.true_positive(actual, pred)
            fn = self.false_negative(actual, pred)
            fp = self.false_positive(actual, pred)
            class_recall = tp / (tp + fn + 1e-6) #adding 1e-6 to denominator to avoid DivisionByZero Error
            class_precision = tp / (tp + fp + 1e-6) #adding 1e-6 to denominator to avoid DivisionByZero Error
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-6) #adding 1e-6 to denominator to avoid DivisionByZero Error
            # keep adding f1 score for all classes
            f1 += class_f1
        # calculate and return average f1 score over all classes
        f1 /= self.num_classes
        return f1
    
    def debug(self, X, y):
        # retrieves the metrics of each individual model for debugging
        debug_metrics=[]
        for i in range(self.num_classes):
            y_debug = [1 if p == i else -1 for p in y]
            accuracy = self.models[i].accuracy_score(X, y_debug)
            precision = self.models[i].precision_score(X, y_debug)
            recall = self.models[i].recall_score(X, y_debug)
            f1_score = self.models[i].f1_score(X, y_debug)
            print(f'digit={i}, accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}')
            debug_metrics.append((i, accuracy, precision, recall, f1_score))
        return debug_metrics