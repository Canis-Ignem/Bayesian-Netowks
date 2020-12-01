#Bayesian Networks
import pomegranate
from pomegranate import BayesianNetwork
from pomegranate import *
#Utils
import seaborn, time
import numpy as np
import matplotlib.pyplot as plt
import copy
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
#Itertools
from itertools import combinations_with_replacement
from itertools import combinations
from itertools import permutations


seaborn.set_style('whitegrid')

numVar = 4
#We load our data and 
cancer_data, target = load_breast_cancer(return_X_y = True)


#This is a very important step, we will store true or false values
#For working with bayesian networks we will try to work with binary values
#We will take as true the values that are higher than the data mean

for i in range( cancer_data.shape[1]):
    m = np.mean(cancer_data[:,i])
    for j in range(cancer_data.shape[0]):
        if cancer_data[j][i] > m:
            cancer_data[j][i] = True
        else:
            cancer_data[j][i] = False

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


tic = time.time()

#Lets compute for each feature the chance of it being true
trues =  []
cont = 0
for i in range(cancer_train_data.shape[1]):

    cont = 0
    for j in range(cancer_train_data.shape[0]):

        if cancer_train_data[j][i]:
            cont += 1

    trues.append(cont/cancer_train_data.shape[0])

#We create the discrete distribution for each feature
feature0 = DiscreteDistribution( {True: trues[0], False: 1-trues[0] } )
feature1 = DiscreteDistribution( {True: trues[1], False: 1-trues[1] } )
feature2 = DiscreteDistribution( {True: trues[2], False: 1-trues[2] } )
feature3 = DiscreteDistribution( {True: trues[3], False: 1-trues[3] } )


#compute all the possible combinations our values can get for our amount of feaures
TRUEFALSE = [True,False]

comb = list(combinations_with_replacement(TRUEFALSE,numVar))

#compute all the possible permutations in order to get all the possible set of values in the dataset
perms = []

for i in comb:
    v = list(permutations(i))
    for j in v:
        if list(j) not in perms:
            perms.append(list(j))


final_train_data = np.concatenate((cancer_train_data,cancer_train_class), axis=1)
FINALTABLE = []
probs = []

#For every set of values we store the row from the dataset in probs
for i in perms:

    for j in final_train_data:
            aux = j[:numVar]
            #print(aux[0])
            if aux[0] == i[0] and aux[1] == i[1] and aux[2] == i[2] and aux[3] == i[3]:
                probs.append(j)

    #For every row in probs we compute the chance of it having True as a class
    cont = 0
    for k in probs:
        if k[4]:
            cont += 1
    #We add to our Table two rows one with the positive class and its chance and once for the negative class
    #with the reverse probability
    negation = copy.copy(i)
    i.append(1)
    negation.append(0)
    i.append(cont/cancer_train_data.shape[0])
    negation.append(1-(cont/cancer_train_data.shape[0]))
    FINALTABLE.append(i)
    FINALTABLE.append(negation)
print("aaaaaaaaaa ", np.shape(FINALTABLE))


cancer_distribution = ConditionalProbabilityTable(FINALTABLE, [feature0,feature1,feature2,feature3])


#Create a state for each feature and one for the table
s0 = State(feature0, name= "feature0")
s1 = State(feature1, name= "feature1")
s2 = State(feature2, name= "feature2")
s3 = State(feature3, name= "feature3")
s4 = State(cancer_distribution, name = "distribution")

#Add the estate to the network
bNetwork = BayesianNetwork("Conditional indepence")
bNetwork.add_states( s0, s1, s2, s3, s4 )

#Add the edges
bNetwork.add_edge(s0,s4)
bNetwork.add_edge(s1,s4)
bNetwork.add_edge(s2,s4)
bNetwork.add_edge(s3,s4)

#for i in FINALTABLE:
#    print(i)

#Finnish the the network
bNetwork.bake()

plt.figure(figsize=(10, 5))
bNetwork.plot()
plt.show()

#Get the probability for a set of values
print(bNetwork.predict_proba( [ { "feature0": True, "feature1": True, "feature2": True, "feature3": False } ] ) )

#Testing the model
apendix = []
for i in range(cancer_test_class.shape[0]):
    apendix.append(None)
apendix = np.array(apendix).reshape(cancer_test_class.shape[0],1)

#Concatenate the missing values to predict
test = np.concatenate((cancer_test_data,apendix), axis=1)

res = bNetwork.predict( test )

print("We took ",time.time()-tic,"seconds building and predicting")

#Extract the classes
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