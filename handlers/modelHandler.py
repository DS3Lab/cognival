from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from keras.activations import relu, linear
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

#TODO write code for cross validation
def modelHandler(X_train, y_train):

    #TODO: change hardcoding of node numbers

    '''
    How many nodes in each hidden layer?
    -midway between input and output
    -less thatn 2x the input nodes
    - 2/3 the input nodes + output nodes
    '''


    model = Sequential()
    model.add(Dense(33, input_dim=50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(22, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()

    '''
    mean_squared_error (mse) and mean_absolute_error (mae): loss functions 

    '''

    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    #TODO: set epoch from the beginnings not hardcoded
    # fit the model
    history = model.fit(X_train, y_train, epochs=400, batch_size=50, verbose=1, validation_split=0.2)

    return history



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
            model.add(Dense(nodes,input_dim=input_dim))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    #TODO: check last layer for multidimensional output
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse',optimizer='adam')
    return model

    #TODO:
    #TODO:change passing parameters for config.json
def modelCV(model_constr,layers,activations,input_dim,batch_size,epochs,X_train,y_train):
    model = KerasRegressor(build_fn=model_constr, verbose=0)

    param_grid = dict(layers=layers,activation=activations,
                      batch_size=batch_size,epochs=epochs,input_dim=input_dim)
    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train,y_train)#Validation_split
    print([grid_result.best_score_,grid_result.best_params_])

def modelPredict(grid,X_test, y_test):
    y_pred = grid.predict(X_test)
    error = y_test - y_pred
    mse = np.mean(np.square(error))
    return mse

def modelHandler():
    #TODO: run 5 fold
    #TODO: return list of 5 items and average on the results [[],[],[],[],[],[]]
    #TODO: save all data into log file 5 fold results
    #TODO: use average final for plot
    pass
