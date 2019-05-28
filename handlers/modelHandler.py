import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #disable tensorflow debugging
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from keras.activations import relu, linear
import keras.backend
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

def create_model(layers, activation, input_dim, output_dim):
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
    print("output dime = "+str(output_dim))
    model.add(Dense(output_dim, activation='linear'))

    #model.summary()
    model.compile(loss='mse',optimizer='adam')

    return model


def modelCV(model_constr,config, X_train,y_train):

    model = KerasRegressor(build_fn=model_constr, verbose=0)

    param_grid = dict(layers=config["layers"], activation=config["activations"], input_dim=[X_train.shape[1]],
                      output_dim=[y_train.shape[1]], batch_size=config["batch_size"], epochs=config["epochs"])

    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='neg_mean_squared_error', cv=config['cv_split'])
    grid_result = grid.fit(X_train,y_train, verbose=0, validation_split=config["validation_split"])

    return grid, grid_result

def modelPredict(grid, words, X_test, y_test):
    #TODO: check predict and results
    y_pred = grid.predict(X_test)
    print("modelPredict")
    print("y_test shape:"+str(y_test.shape))
    print("y_shape :"+str(y_pred.shape))
    if y_test.shape[1] ==1:
        print("univariate model ")
        y_pred = y_pred.reshape(-1,1)
        print("y_shape :" + str(y_pred.shape))
    error = y_test - y_pred
    print("error")
    print(error)
    word_error = np.hstack([words,error])
    print(word_error)
    print("word error shape "+str(word_error.shape))
    if word_error.shape[1] ==1:
        mse = np.mean(np.square(error))
    else:
        mse = np.mean(error,axis=0)
    return mse, word_error

def modelHandler(config,words_test, X_train, y_train, X_test, y_test):
    grids =[]
    grids_result = []
    mserrors = []
    word_error = np.array(['word','error'],dtype='str')
    for i in range(len(X_train)):
        grid, grid_result = modelCV(create_model,config,X_train[i],y_train[i])
        grids.append(grid)
        grids_result.append(grid_result)
        mse, w_e = modelPredict(grid,words_test[i],X_test[i],y_test[i])
        mserrors.append(mse)
        word_error = np.vstack([word_error,w_e])
    #TODO: CHECK OUTPUT and results
    return word_error, grids_result, mserrors

