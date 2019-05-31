
import pandas as pd

def headerGen(fileName):
	df = pd.read_csv(fileName,sep=" ")
	print(df.shape)
	dim = df.shape[1]
	header = "word"
	for i in range(1,dim):
		header = header+" x"+str(i)
	return header


def main():
	fileName = "/datasets/embeddings/bert/bert_large_all.txt"
	print(headerGen(fileName))

if __name__=="__main__":
	main()
