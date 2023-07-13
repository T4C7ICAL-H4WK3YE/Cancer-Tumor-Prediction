import pandas as pd
import numpy as np

ratio = 0.67  #training to testing ratio
label_map = {'B': -1.0, 'M': 1.0}	#converting to integer
dataset=pd.read_csv("Dsata Set for Assignment 1.csv",usecols=range(1,32),dtype={'diagnosis': 'category'}).replace({'diagnosis': label_map}).drop([0])

dataset=dataset.to_numpy()

total_rows=dataset.shape[0]     #getting total number of tuples
#print(total_rows)
train_size=int(total_rows*ratio)
rng = np.random.RandomState(8) #random number generator
rng.shuffle(dataset)
dataset_train=dataset[0:train_size]
dataset_test=dataset[train_size:]
#finding mean of each column
mean_list=list(np.nanmean(dataset_train,axis=0))
#print(mean_list)
#print(dataset[527][0])
# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i+1]
	return 1.0 if activation >= 0.0 else -1.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train,n_epoch):
	
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		y=0	
		for row in train:
			row=list(row)
			row=[mean_list[row.index(x)] if x!=x else x for x in row]
			prediction = predict(row, weights)
			if(prediction!=row[0]):

			#error = row[0] - prediction
			#sum_error += error**2
				weights[0] = weights[0] + row[0]
				for i in range(len(row)-1):
					weights[i + 1] = weights[i + 1] + row[0] * row[i+1]
				
	#print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))	
		#arr[epoch]=sum_error
		
		#if(epoch>0) and (arr[epoch]==arr[epoch-1]):
		#	count+=1
			
		#if(count>5):
			
			#break
	return weights

def test_weights(test,final_weight):
	total=0
	correct=0
	for row in test:
		total+=1
		row=list(row)
		row=[mean_list[row.index(x)] if x!=x else x for x in row]
		if(predict(row,final_weight)==row[0]):
			correct+=1
	accuracy=correct/total*100
	return accuracy		
# Calculate weights
# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# [7.673756466,3.508563011,1]]    
n_epoch =1000
weights = train_weights(dataset_train, n_epoch)
accuracy= test_weights(dataset_test,weights)
print("Accuracy is:"+str(accuracy)+"%")
print("Weight vector:")
print(weights)