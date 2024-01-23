
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


"""
WeatherClassifier: class used to train/test a MLP Classifier based on the training and test sets provided.

""" 
class WeatherClassifier:
    
    def __init__(self, X:np.ndarray, y: np.ndarray,random_seed=1, max_iter=200):
        
        """
        Creates a weather classifier.
    
        Args:
            X (numpy.ndarray): training set having the descriptor for each image on each row
            y (numpy.ndarray): array of the image labels ordered according to X
            random_seed (int): random seed to be used to achieve invariance accross multiple executions
            max_iter (int): maximum number of iteration to train the classifier

        """ 

        #checking the parameters
        if( (X is None) or (y is None)):
            raise Exception('X or y are none')


        if(random_seed<=0):
            raise Exception('Wrong random seed provided, must be grater than 0')
        elif(max_iter<=0):
            raise Exception('Wrong maximum number of iterations (max_iter) provided, must be grater than 0')


        self.X=X
        self.y=y

        #creating a standard scaler to scale the features
        self.scaler = StandardScaler().fit(self.X)

        #creating a MLP classifier with the parameters provided
        self.mlp_classifier=MLPClassifier(random_state=random_seed,max_iter=max_iter)

        self.random_seed=random_seed
        self.max_iter=max_iter

        #setting random seeds
        np.random.seed(self.random_seed)




    def permute(self,X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """
        Performs a random permutation on the provided data.
    
        Args:
            X (numpy.ndarray): dataset having the descriptor for each image on each row
            y (numpy.ndarray): array of the image labels ordered according to X
        Returns:
            X_perm (numpy.ndarray): permuted verision of X
            y_perm (numpy.ndarray): permuted version of y

        """ 
        if( (X is None) or (y is None)):
            raise Exception('X or y are none')

        #creating a permutation
        permutation = np.random.permutation(X.shape[0])

        #permuting data
        X_perm = X[permutation]
        y_perm = y[permutation]
        return X_perm,y_perm


    def scale(self,X: np.ndarray) -> np.ndarray:
        """
        Scales data according to the training set .
    
        Args:
            X (numpy.ndarray): training set having the descriptor for each image on each row
        Returns:
            X_scaled (numpy.ndarray): scaled verision of X

        """

        if(X is None):
            raise Exception('X is none')

        #scaling data using the scaler previously fitted on the training data
        X_scaled = self.scaler.transform(X)

        return X_scaled


    def trainClassifier(self) -> MLPClassifier:
        """
        Trains the MLPClassifier according to the training set .
    
        Returns:
            mlp_classifier (MLPClassifier): trained classifier

        """

        
        self.mlp_classifier=MLPClassifier(random_state=self.random_seed,max_iter=self.max_iter)

        #permuting and scaling data
        X_perm,y_perm=self.permute(self.X,self.y)
        X_perm_scaled=self.scale(X_perm)

        #training the classifier
        self.mlp_classifier.fit(X_perm_scaled,y_perm)

        return self.mlp_classifier



    def score(self,X_test: np.ndarray, y_test:np.ndarray) -> float:
        """
        Test the provided dataset on the MLPClassifier and returns the score.
    
        Returns:
            score (float): the score obtained with the provided dataset

        """

        #scale data
        X_test_scaled=self.scale(X_test)

        return self.mlp_classifier.score(X_test_scaled,y_test)



    def gridSearch(self) -> tuple[MLPClassifier,float]:
        """
        Trains the MLPClassifier according to the training set, exploiting grid search to find the best parameter combination.
    
        Returns:
            best_estimator_ (MLPClassifier): trained classifier (with the best parameter combination)
            best_score_ (float): best classifier score
        """

        classifier=MLPClassifier()

        #permuting and scaling data
        X_perm,y_perm=self.permute(self.X,self.y)
        X_perm_scaled=self.scale(X_perm)


        #creating a grid of parameter to test different layers/neuron combinations
        param_grid = {'hidden_layer_sizes': [(10,),(20,),(100,),(10,10),(100,100)],
                    'solver': ['lbfgs', 'adam'], 
                    'random_state': [self.random_seed],
                    'max_iter': [self.max_iter]
                    }
        
        #performing grid search
        grid = GridSearchCV(classifier, param_grid=param_grid, cv=5, verbose=4)
        grid.fit(X_perm_scaled, y_perm)

        self.mlp_classifier=grid.best_estimator_

        return grid.best_estimator_ ,grid.best_score_

        
        









