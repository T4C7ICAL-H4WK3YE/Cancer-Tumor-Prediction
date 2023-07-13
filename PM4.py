import pandas as pd
import numpy as np

ratio = 0.67  #training to testing ratio
label_map = {'B': -1.0, 'M': 1.0}	#converting to integer
dataset=pd.read_csv("Dsata Set for Assignment 1.csv",usecols=range(2,32),dtype={'diagnosis': 'category'}).replace({'diagnosis': label_map}).drop([0])
Y_vector=pd.read_csv("Dsata Set for Assignment 1.csv",usecols=[1],dtype={'diagnosis': 'category'}).replace({'diagnosis': label_map}).drop([0])
dataset=dataset.to_numpy()
Y_vector=Y_vector.to_numpy()
total_rows=dataset.shape[0]
#print(Y_vector[19][0])
dataset=np.transpose(dataset)
rng = np.random.RandomState(33) #random number generator
rng.shuffle(dataset)          # shuffling the rows
dataset=np.transpose(dataset)#columns shuffled of original dataset

train_size=int(total_rows*ratio)
dataset_train=dataset[0:train_size]
dataset_test=dataset[train_size:]
#finding mean of each column
mean_list=list(np.nanmean(dataset_train,axis=0))

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else -1.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train,n_epoch):
	
	weights = [0.0 for i in range(len(train[0])+1)]
	for epoch in range(n_epoch):
		arr=[0.0 for i in range(n_epoch)]
		sum_error = 0.0
		y=0	
		for row in train:
			row=list(row)
			row=[mean_list[row.index(x)] if x!=x else x for x in row]
			prediction = predict(row, weights)
			if(prediction!=Y_vector[y][0]):

			#error = row[0] - prediction
			#sum_error += error**2
				weights[0] = weights[0] + Y_vector[y][0]
				for i in range(len(row)):
					weights[i + 1] = weights[i + 1] + Y_vector[y][0] * row[i]
			y+=1	
	#print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))	
		#arr[epoch]=sum_error
		
		#if(epoch>0) and (arr[epoch]==arr[epoch-1]):
		#	count+=1
			
		#if(count>5):
			
			#break


	return weights

def test_weights(test,final_weight,train_size):
	total=0
	correct=0
	for row in test:
		total+=1
		row=list(row)
		row=[mean_list[row.index(x)] if x!=x else x for x in row]
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