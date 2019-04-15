

def modelHandler():
    model = Sequential()
    model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()

    '''
    mean_squared_error (mse) and mean_absolute_error (mae): loss functions 

    '''

    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    # fit the model
    history = model.fit(X_train, y_train, epochs=300, batch_size=50, verbose=1, validation_split=0.2)

    pass

def prefiction():
    # prediction
    Xnew = np.array([[40, 0, 26, 9000, 8000]])
    Xnew = scaler_x.transform(Xnew)
    ynew = model.predict(Xnew)
    # invert normalize
    ynew = scaler_y.inverse_transform(ynew)
    Xnew = scaler_x.inverse_transform(Xnew)
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))


    pass