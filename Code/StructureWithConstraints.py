#Bayesian Networks
import pomegranate
from pomegranate import BayesianNetwork
#Utils
import seaborn, time
import numpy as np
import matplotlib.pyplot as plt
#Data
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
#Accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import random
import networkx
from pomegranate.utils import plot_networkx

seaborn.set_style('whitegrid')

#Number of variables we are going to use
numVar = 4

#We load our data and 
cancer_data, target = load_breast_cancer(return_X_y = True)


#This is a very important step, we will store true or false values
#For working with bayesian networks we will try to work with binary values
#We will take as true the values that are higher than the data mean
for i in range( cancer_data.shape[1]):
    cancer_data[:,i] =cancer_data[:,i] > np.mean(cancer_data[:,i])

#Lets add the y so our bayesian model can predict it
#target = target > np.mean(target)

#We also need to reshape to make the concatenation in a correct maner with the data
target = np.reshape(target,(target.shape[0],1))
print("Class target shape: ",target.shape)


#We will take the best 4 features
fSelector = SelectKBest(chi2, k= numVar)
selected_data = fSelector.fit_transform(cancer_data,target)
print("Feature selected data shape: ",selected_data.shape)


# We could use even indices for train data
cancer_train_data = selected_data [0::2]
cancer_train_class = target[0::2]

# We use odd could indices for test data
cancer_test_data  = selected_data[1::2]
cancer_test_class = target[1::2]

#We generate an appendix of none for the model to predict
apendix = []
for i in range(cancer_test_class.shape[0]):
    apendix.append(None)
apendix = np.array(apendix).reshape(cancer_test_class.shape[0],1)

t1 = time.time()
#We learn how the variables influence eachother
model = BayesianNetwork.from_samples(cancer_train_data, algorithm='exact') #A* algorithm for score based struture learning
plt.figure(figsize=(6, 5))
model.plot()
plt.show()

#We will draw a new graph based on the influence of each variable on the other and we will append to all the nodes the class node
g = networkx.DiGraph()
b = tuple([numVar])
for i in range(len(model.structure)):
    if len(model.structure[i]) > 0:
        g.add_edge(model.structure[i],tuple([i]))
        g.add_edge(tuple([i]),b)


#contraint graph
plot_networkx(g)
plt.show()

#We add the class to our training datas
final_train_data = np.concatenate((cancer_train_data,cancer_train_class), axis=1)
model = BayesianNetwork.from_samples(final_train_data, algorithm='exact',constraint_graph= g) #A* algorithm for score based struture learning
plt.figure(figsize=(5, 5))
model.plot()
plt.show()

'''
#Inserting random Nones
for i in range(cancer_test_data.shape[0]):
    for j in range(len(cancer_test_data[i])):
        if random.randint(1,5) == 4:
            cancer_test_data[i][j] = None
'''
#We will append an array of nones to where the class should be and try and predict it
test = np.concatenate((cancer_test_data,apendix), axis=1)
res = model.predict( test )

print("The time spent building the hole model and getting the preddictions was: ", time.time()-t1)
#Since the models gives back all the values we select only the class value
predict = []
k = np.shape(res)
for i in range(k[0]):
    predict.append(res[i][numVar])
predict = np.array(predict).reshape(k[0],1)

#We check how were our results
acc = accuracy_score(predict, cancer_test_class)

confusionMTRX = confusion_matrix(cancer_test_class,predict)
print(acc)
print(confusionMTRX)
