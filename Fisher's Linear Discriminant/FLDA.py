#imported the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LDA:

  def __init__(self,n_components):

    self.n_components = n_components #components -> Features
    self.linear_discriminants = None

  def fit(self,X,y):

    n_features = X.shape[1] #number of features = number of elements in each column of X
    class_labels = np.unique(y) #assigns unique values of y to class_labels
    n_classes = class_labels.size # number of classes in y

    mean_overall = np.mean(X, axis = 0) #mean along columns of X
    #Create S_W and S_B matrices of respective dimensions with all entries initalized to 0
    S_W = np.zeros((n_features,n_features))
    S_B = np.zeros((n_features,n_features))
    #Empty list which is filled with respective class (of y) associated X's (feature vectors)
    X_c = [];
    for c in class_labels:
      for j in range(y.size): #y.size -> number of elements in y
            if y[j,0] == c:
               if len(X_c)==0:
                  X_c = X[j]
               else:
                  X_c = np.vstack((X_c, X[j])) #np.vstack () is used to stack arrays vertically
      
      mean_c = np.mean(X_c, axis = 0) #Calculate class-wize mean of X
      S_W += (X_c - mean_c).T.dot(X_c - mean_c)

      n_c = X_c.shape[0] #n_c = number of rows in X_c

      mean_diff = (mean_c - mean_overall).reshape(n_features,1)

      S_B += n_c * (mean_diff).dot(mean_diff.T)

      A = np.linalg.inv(S_W).dot(S_B) #Calculate [(S_W)^-1]*(S_B)
      eigenvectors,eigenvalues = np.linalg.eig(A) #Finding eigenvectors and eigenvalues of A
      eigenvectors = eigenvectors.T
      idxs = np.argsort(abs(eigenvalues))[::-1] #np.argsort() with [::-1] gives list with indices of abs(eigenvalues) in an order according to which they would have been in descending order
      eigenvalues = eigenvalues[idxs]
      eigenvectors = eigenvectors[idxs]
      self.linear_discriminants = eigenvectors[0:self.n_components]

  def transform(self, X):
    #Project data
    return np.dot(X, self.linear_discriminants.T)

label_map = {'B': 0, 'M': 1} #Mapping B to 0 and M to 1

#Read csv and split data into x and y
x = pd.read_csv("Dsata Set for Assignment 1.csv",skiprows=[0], usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
x = x.to_numpy() #Added this
for d in range(30):
  #access the dth column (index (d-1)) and replace nan with mean
  x[:, d] = np.nan_to_num(x[:, d], nan= np.nanmean(x[:,d]))

y = pd.read_csv("Dsata Set for Assignment 1.csv", usecols=[1], dtype={'diagnosis': 'category'}).replace({'diagnosis': label_map}).drop([0]) 
y = y.to_numpy()
#access the 1st column (index 0) and replace nan with mean
y[:, 0] = np.nan_to_num(y[:, 0], nan= np.nanmean(y[:,0]))

# Select Train ratio
ratio = 0.67

total_xrows = x.shape[0]
train_xsize = int(total_xrows*ratio)
total_yrows = y.shape[0]
train_ysize = int(total_yrows*ratio)
Avg_Accuracy = [0,0]
Total_Accuracy = [0,0]
for b in range(2):
  x = np.transpose(x)
  np.random.shuffle(x)
  x = np.transpose(x)
  #split features into train and test
  rng = np.random.RandomState(42)
  #Created for loop and adjusted indentation from x_train onwards
  for a in range(10):
    rng.shuffle(x)
    rng.shuffle(y)
    #split labels into train and test
    x_train = x[0:train_xsize]
    x_test = x[train_xsize:]
    y_train = y[0:train_ysize]
    y_test = y[train_ysize:]

    y_train.shape[0]
    ld = LDA(x_train.shape[1])
    ld.fit(x_train,y_train)
    X_projected = ld.transform(x_train)
    #Projected
    x1 = X_projected[:, 0] #The result of X_projected[:, 0] is a one-dimensional array that contains all the elements from the first column of X_projected
    x2 = X_projected[:, 1]

    mean_x1 = np.mean(x1)
    mean_x2 = np.mean(x2)
    # total no.of input values
    count = len(x1)

    # using the formula to calculate m & c of decision boundary of projected points
    numer = 0
    denom = 0
    for i in range(count):
      numer += (x1[i] - mean_x1) * (x2[i] - mean_x2)
      denom += (x1[i] - mean_x1) ** 2
    m_p = numer / denom
    c_p = mean_x2 - (m_p * mean_x1)
    max_x = np.max(x1)
    min_x = np.min(x2)

    # calculating line values x and y
    x_p = np.linspace (min_x, max_x,1000)
    y_p = c_p + m_p * x_p

    # plotting regression line
    plt.plot(x_p, y_p, color='#58b970', label='Regression Line')
    # Once we get regression line, we don't need training values. We use test values to check for accuracy wrt line obtained
    X1_Projected =  ld.transform(x_test)
    x3 = X1_Projected[:, 0]
    x4 = X1_Projected[:, 1]
    # plotting points
    plt.scatter(x3, x4, c= y_test, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis',2))
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.colorbar()
    plt.show()
    threshold = np.mean(X1_Projected)
    y_pred = (X1_Projected > threshold).astype(int) #convert to bool 0 or 1 whether condition is satisfied or not
    
    # Calculate the misclassification rate
    misclassification_rate = np.mean(y_pred != y_test)
    Accuracy = (1-misclassification_rate)*100
    print("Accuracy:", Accuracy)
    Total_Accuracy[b] += Accuracy
  #Accuracy calculation for each Engineering Task
  Avg_Accuracy[b] = Total_Accuracy[b]/10
  print("\nAvg Accuracy is:",Avg_Accuracy[b])
  print("\nEnd of Engineering Task.") #Now we perform Second Engineering Task by considering a random permutation of feature tuples
#End of Task
print("End of Task")
