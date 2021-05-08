import numpy as np

from sklearn.model_selection import GridSearchCV,cross_val_score
from ray.tune.sklearn import TuneGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype

import time
import argparse

import ray


parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='6379')



def get_data(random_state, test_size = 0.2):
    data = fetch_covtype(download_if_missing=True, shuffle=True,random_state=random_state)
    
    X = data['data']
    y = data['target']
    n_train = int((1-test_size)*X.shape[0])
    return X[:n_train],y[:n_train],X[n_train:],y[n_train:]

def train_default(random_state = 42):
    X_train,y_train,X_test,y_test = get_data(random_state=random_state)

    default_model = RandomForestClassifier(random_state=random_state)

    default_model.fit(X_train,y_train)

    accuracy = np.mean(default_model.predict(X_test)==y_test)

    params = default_model.get_params()

    return {'accuracy':accuracy,'params':params}

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def train_search(n_estimators,max_depth,ccp_alpha,X,y,random_state = 42):
    ''' max_depth, n_estimators and ccp_alpha '''
    
    target_model = RandomForestClassifier(n_estimators = n_estimators,max_depth=max_depth,ccp_alpha=ccp_alpha,random_state=random_state)
    
    score = cross_val_score(target_model, X, y,scoring='accuracy',cv=4)

    return (score,target_model)

def train_grid_search(random_state = 42):
    ''' max_depth, n_estimators and ccp_alpha '''
    X_valid,y_valid,_,_ = get_data(random_state=random_state,test_size=0.1)
    
    n_estimators = [10,50]
    
    max_depth = [5,50,100]
    
    ccp_alpha = [0.001,0.01]
    
    hyperparameter_grid = {'n_estimators':n_estimators,'max_depth':max_depth,'ccp_alpha': ccp_alpha}
    
    target_model = RandomForestClassifier(random_state=random_state)
    
    # create search object
    grid_cv = TuneGridSearchCV(estimator=target_model,
                               param_grid=hyperparameter_grid,
                               cv=4)
    
    # Fit on the all training data using random search object
    grid_cv.fit(X_valid, y_valid)
    return grid_cv.best_estimator


if __name__ == '__main__':
    args = parser.parse_args()
    
    ray.init(address=f'{args.ip}:{args.port}',_redis_password='0520')
    
    #default_result = train_default()
    
    t_start = time.time()
    
    estimator = train_grid_search()

    t_end = time.time()
    
    X_train,y_train,X_test,y_test = get_data(random_state=44,test_size=0.2)
    
    estimator.fit(X_train,y_train)
    
    accuracy = np.mean(estimator.predict(X_test)==y_test)
    
    grid_result = {'accuracy':accuracy,'best_params':estimator.get_params()}

    print(f'Grid Search Time:{t_end-t_start}')
    print(f'Grid Search Result:{grid_result}')
   
