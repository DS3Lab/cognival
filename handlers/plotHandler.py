import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plotHandler(title, history, outputDir):

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(outputDir+"/"+title+ '.png')
    plt.clf()
    #plt.show()
    pass