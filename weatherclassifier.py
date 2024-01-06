
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class WeatherClassifier:
    
    def __init__(self, X:np.ndarray, y: np.ndarray,random_seed=1, max_iter=200):
        if( (X is None) or (y is None)):
            raise Exception('X or y are none')


        if(random_seed<=0):
            raise Exception('Wrong random seed provided, must be grater than 0')
        elif(max_iter<=0):
            raise Exception('Wrong maximum number of iterations (max_iter) provided, must be grater than 0')

        self.X=X
        self.y=y
        self.scaler = StandardScaler().fit(self.X)
        self.mlp_classifier=MLPClassifier(random_state=random_seed,max_iter=max_iter)
        self.random_seed=random_seed
        self.max_iter=max_iter
        np.random.seed(self.random_seed)




    def permute(self,X: np.ndarray, y: np.ndarray):
        if( (X is None) or (y is None)):
            raise Exception('X or y are none')

        permutation = np.random.permutation(X.shape[0])
        X_perm = X[permutation]
        y_perm = y[permutation]
        return X_perm,y_perm


    def scale(self,X: np.ndarray):
        if(X is None):
            raise Exception('X is none')

        X_scaled = self.scaler.transform(X)

        return X_scaled


    def train_classifier(self):

        self.mlp_classifier=MLPClassifier(random_state=self.random_seed,max_iter=self.max_iter)
        X_perm,y_perm=self.permute(self.X,self.y)
        X_perm_scaled=self.scale(X_perm)
        self.mlp_classifier.fit(X_perm_scaled,y_perm)

        return self.mlp_classifier



    def score(self,X_test: np.ndarray, y_test:np.ndarray):

        X_test_scaled=self.scale(X_test)

        return self.mlp_classifier.score(X_test_scaled,y_test)



    def grid_search(self):

        classifier=MLPClassifier()

        X_perm,y_perm=self.permute(self.X,self.y)
        X_perm_scaled=self.scale(X_perm)

        param_grid = {'hidden_layer_sizes': [(10,),(20,),(100,),(10,10),(100,100)],
                    'solver': ['lbfgs', 'adam'], 
                    'random_state': [self.random_seed],
                    'max_iter': [self.max_iter]
                    }
        
        grid = GridSearchCV(classifier, param_grid=param_grid, cv=5, verbose=4)
        grid.fit(X_perm_scaled, y_perm)

        self.mlp_classifier=grid.best_estimator_

        return grid.best_estimator_ ,grid.best_score_

        
        









