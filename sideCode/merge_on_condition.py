import pandas as pd
import numpy as np

def update(df1,df2,on_column,whole_row):
	 # Both dataframes have to have same column names
	header = list(df1)
	header = header[1:]

	start = df1.shape[1]
	to_update = df1.merge(df2,on=on_column,how='left').iloc[:,start:].dropna()
	to_update.columns = header

	if whole_row:
		#UPDATE whole row when NaN appears
		df1.loc[df1[header[0]].isnull(),header] = to_update
	else:
		#UPDATE just on NaN values
	 	for elem in header:
	 		df1.loc[df1[elem].isnull(),elem] = to_update[elem]

	return df1



def main():
	df = pd.DataFrame({'A':[1,2,3,4],'B':[np.NaN,500,np.NaN,np.NaN],'C':[np.NaN,7.0,np.NaN,np.NaN]})
	df_new = pd.DataFrame({'A':[1,5,3],'B':[4,500,8],'C':[0,0,0]})
	df_up = pd.DataFrame({'B':[0,0,0],'C':[0,0,0]})
	print(df)
	print(df_new)
	update(df, df_new, 'A')
	print(df)
	pass



if __name__=="__main__":
	main()