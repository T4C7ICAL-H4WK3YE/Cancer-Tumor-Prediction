import pandas as pd
import numpy as np

ratio = 0.67  #training to testing ratio
label_map = {'B': -1.0000000, 'M': 1.0000000}	#converting to integer
dataset=pd.read_csv("Dsata Set for Assignment 1.csv",usecols=range(2,32)).drop([0])
Y_vector=pd.read_csv("Dsata Set for Assignment 1.csv",usecols=[1],dtype={'diagnosis': 'category'}).replace({'diagnosis': label_map}).drop([0])
dataset=dataset.to_numpy()
Y_vector=Y_vector.to_numpy()
Normalized_data = (dataset - np.nanmean(dataset, axis=0))/np.nanstd(dataset, axis=0)
#print(Normalized_data[527])

total_rows=dataset.shape[0]
#print(total_rows)
train_size=int(total_rows*ratio)
dataset_train=Normalized_data[0:train_size]
dataset_test=Normalized_data[train_size:]
#finding mean of each column
mean_list=list(np.nanmean(dataset_train,axis=0))
std_list=list(np.nanstd(dataset_train,axis=0))
#print(pd.isna(dataset[533][0]))
# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else -1.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train,n_epoch):
	
	weights = [0.0000000 for i in range(len(train[0])+1)]
	for epoch in range(n_epoch):
		y=0	
		for row in train:
			row=list(row)
			#row=[0 if pd.isna(row[x]) else ((row[x]-mean_list[x])/std_list[x]) for x in range(len(row))]
			row=[0.0 if x!=x else x for x in row]
			prediction = predict(row, weights)
			if(prediction!=Y_vector[y][0]):
				weights[0] = weights[0] + Y_vector[y][0]
				for i in range(len(row)):
					weights[i + 1] = weights[i + 1] + Y_vector[y][0] * row[i]
			y+=1	

	return weights

def test_weights(test,final_weight,train_size):
	total=0
	correct=0
	for row in test:
		total+=1
		row=list(row)
		row=[0.0 if x!=x else x for x in row]
		if(predict(row,final_weight)==Y_vector[train_size][0]):
			correct+=1
		train_size+=1	
	accuracy=correct/total*100
	return accuracy		
   
n_epoch = 1000
weights = train_weights(dataset_train, n_epoch)
accuracy= test_weights(dataset_test,weights,train_size)
print("Accuracy is:"+str(accuracy)+"%")
print("Weight vector:")
print(weights)