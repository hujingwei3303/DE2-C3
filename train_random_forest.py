import numpy as np
from ray.tune.sklearn import TuneGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
import time
import ray
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default='')
parser.add_argument('--port', type=str, default='6379')



def get_data(random_state, test_size = 0.2):
    data = fetch_covtype(download_if_missing=True, shuffle=True,random_state=random_state)
    X = data['data'][:5000]
    y = data['target'][:5000]
    n_train = int((1-test_size)*X.shape[0])
    return X[:n_train],y[:n_train],X[n_train:],y[n_train:]

def train_default(random_state = 42):
    X_train,y_train,X_test,y_test = get_data(random_state=random_state)

    default_model = RandomForestClassifier(random_state=random_state)

    default_model.fit(X_train,y_train)

    accuracy = np.mean(default_model.predict(X_test)==y_test)

    params = default_model.get_params()

    return {'accuracy':accuracy,'params':params}

def train_grid_search(random_state = 42):
    ''' max_depth, n_estimators and ccp_alpha '''
    X_train,y_train,X_test,y_test = get_data(random_state=random_state,test_size=0.1)
    
    n_estimators = list(range(50,1050,100))
    
    max_depth = list(range(2,15,3))
    
    max_depth.insert(0,None)
    
    ccp_alpha = np.arange(0.,0.02,0.005)
    
    hyperparameter_grid = {'n_estimators': n_estimators,'max_depth': max_depth,'ccp_alpha': ccp_alpha}
    
    target_model = RandomForestClassifier(random_state=42)
    
    # create search object
    grid_cv = TuneGridSearchCV(estimator=target_model,
                               param_grid=hyperparameter_grid,
                               cv=5, 
                               scoring = 'accuracy')
    
    # Fit on the all training data using random search object
    grid_cv.fit(X_train, y_train)
    
    accuracy = np.mean(grid_cv.predict(X_test)==y_test)
    
    params = grid_cv.estimator_.get_params()

    return {'best_accuracy':accuracy,'best_params':params}

if __name__ == '__main__':
    args = parser.parse_args()
    
    ray.init(redis_address=f'{args.ip}:{args.port}')
    
    default_result = train_default()
    
    t_start = time.time()
    grid_result = train_grid_search()
    t_end = time.time()
    
    with open("training_log.log") as f:
    	f.write(f'Train Default:{default_result}')
	f.write('\n')
    	f.write(f'Grid Search Time:{t_end-t_start}')
	f.write('\n')
    	f.write(f'Grid Search Result:{grid_result}')
	f.write('\n')
    