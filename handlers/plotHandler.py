import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plotHandler(history, version, outputDir):


    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(outputDir+"/"+str(version)+ '.png')
    plt.show()


    pass