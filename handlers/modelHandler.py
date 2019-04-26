from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from keras.activations import relu, linear
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

def create_model(layers, activation, input_dim):
    '''

    :param layers: [hiddenlayer1_nodes,hiddenlayer2_nodes,...]
    :param activation: relu
    :param input_dim: number_of_input_nodes
    :return: model
    '''
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=input_dim, activation=activation))
        else:
            model.add(Dense(nodes, activation=activation))
    #TODO: check last layer for multidimensional output
    model.add(Dense(1, activation='linear'))

    model.summary()
    model.compile(loss='mse',optimizer='adam')

    return model


def modelCV(model_constr,config, X_train,y_train):

    model = KerasRegressor(build_fn=model_constr, verbose=0)
    #TODO: change config['cv'][""layers] and add optimal option and save optimal option to config.json
    param_grid = dict(layers=config["layers"], activation=config["activations"],input_dim=[X_train.shape[1]],
                      batch_size=config["batch_size"], epochs=config["epochs"])

    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train,y_train, verbose=1, validation_split=0.2)#Validation_split

    return grid, grid_result

# def modelPredict(grid,X_test, y_test):
#     y_pred = grid.predict(X_test)
#     error = y_test - y_pred
#     mse = np.mean(np.square(error))
#     return mse

def modelHandler(config, X_train, y_train):
    #TODO: run 5 fold
    #TODO: return list of 5 items and average on the results [[],[],[],[],[],[]]
    #TODO: save all data into log file 5 fold results
    #TODO: use average final for plot
    grid, grid_result = modelCV(create_model,config,X_train,y_train)
    print([grid_result.best_score_, grid_result.best_params_])

    return grid_result.best_estimator_.model.history, grid_result.best_params_
