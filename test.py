from eyetrackingHandler import eyetrackingHandler

#TODO: don't hardcode all inputs, create command line interface
#cognitiveData = "../datasets/eeg/zuco/zuco_scaled.txt"
cognitiveData ='../datasets/dundee/dundee_scaled.txt'
#TODO: SET to choose for specific dimension
wordEmbDir = "../datasets/glove-6B/glove.6B.50d.copy.txt"

def run():

    eyetrackingHandler(cognitiveData,wordEmbDir)

    pass


def main():

    run()

    pass


if __name__ == "__main__":
    main()