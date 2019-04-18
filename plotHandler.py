import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plotHanlder(history):


    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    pass