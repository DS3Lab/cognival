from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

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

# def prefiction():
#     # prediction
#     Xnew = np.array([[40, 0, 26, 9000, 8000]])
#     Xnew = scaler_x.transform(Xnew)
#     ynew = model.predict(Xnew)
#     # invert normalize
#     ynew = scaler_y.inverse_transform(ynew)
#     Xnew = scaler_x.inverse_transform(Xnew)
#     print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
#
#
#     pass