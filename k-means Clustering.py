import pandas as pd
import numpy

import warnings
warnings.filterwarnings("ignore")

def class1_class2():
    traindataset=pd.read_csv(r'.\train.csv' )
    testdataset=pd.read_csv(r'.\test.csv')
    indexNametrain = traindataset[ (traindataset['Class'] == "class-3")].index#selects class-3 from tain data
    indexNametest = testdataset[ (testdataset['Class'] == "class-3")].index#selects class-3 from  test data
    traindataset.drop(indexNametrain , inplace=True)#removes class-3 from the train data
    testdataset.drop(indexNametest , inplace=True)#removes class-3 from test data
    #assign numerical values to the classes
    traindataset.Class[traindataset.Class == 'class-1'] = 1
    traindataset.Class[traindataset.Class == 'class-2'] = -1
    testdataset.Class[testdataset.Class == 'class-1'] = 1
    testdataset.Class[testdataset.Class == 'class-2'] = -1
    #calling the functions
    X_train,Y_train,X_test,Y_test = separate_data(traindataset,testdataset)
    predict(X_train,Y_train,X_test,Y_test)

def class2_class3():
    traindataset=pd.read_csv(r'.\train.csv' )
    testdataset=pd.read_csv(r'.\test.csv')
    indexNametrain = traindataset[ (traindataset['Class'] == "class-1")].index
    indexNametest = testdataset[ (testdataset['Class'] == "class-1")].index
    traindataset.drop(indexNametrain , inplace=True)
    testdataset.drop(indexNametest , inplace=True)
    traindataset.Class[traindataset.Class == 'class-2'] = 1
    traindataset.Class[traindataset.Class == 'class-3'] = -1
    testdataset.Class[testdataset.Class == 'class-2'] = 1
    testdataset.Class[testdataset.Class == 'class-3'] = -1
    X_train,Y_train,X_test,Y_test = separate_data(traindataset,testdataset)
    predict(X_train,Y_train,X_test,Y_test)
    
def class1_class3():
    traindataset=pd.read_csv(r'.\train.csv' )
    testdataset=pd.read_csv(r'.\test.csv')
    indexNametrain = traindataset[ (traindataset['Class'] == "class-2")].index
    indexNametest = testdataset[ (testdataset['Class'] == "class-2")].index
    traindataset.drop(indexNametrain , inplace=True)
    testdataset.drop(indexNametest , inplace=True)
    traindataset.Class[traindataset.Class == 'class-1'] = 1
    traindataset.Class[traindataset.Class == 'class-3'] = -1
    testdataset.Class[testdataset.Class == 'class-1'] = 1
    testdataset.Class[testdataset.Class == 'class-3'] = -1
    X_train,Y_train,X_test,Y_test = separate_data(traindataset,testdataset)
    predict(X_train,Y_train,X_test,Y_test) 
    

def class1_all():
    traindataset=pd.read_csv(r'.\train.csv' )
    testdataset=pd.read_csv(r'.\test.csv')
    #assign numerical values to each class according the seperation
    traindataset.Class[traindataset.Class == 'class-1'] = 1
    traindataset.Class[traindataset.Class == 'class-2'] = -1
    traindataset.Class[traindataset.Class == 'class-3'] = -1
    testdataset.Class[testdataset.Class == 'class-1'] = 1
    testdataset.Class[testdataset.Class == 'class-2'] = -1
    testdataset.Class[testdataset.Class == 'class-3'] = -1

    X_train,Y_train,X_test,Y_test = separate_data(traindataset,testdataset)
    predict(X_train,Y_train,X_test,Y_test) 
    l2norm(X_train,Y_train,X_test,Y_test,traindataset,testdataset)
        
def class2_all():
    traindataset=pd.read_csv(r'.\train.csv' )
    testdataset=pd.read_csv(r'.\test.csv')
    
    traindataset.Class[traindataset.Class == 'class-1'] = -1
    traindataset.Class[traindataset.Class == 'class-2'] = 1
    traindataset.Class[traindataset.Class == 'class-3'] = -1
    testdataset.Class[testdataset.Class == 'class-1'] = -1
    testdataset.Class[testdataset.Class == 'class-2'] = 1
    testdataset.Class[testdataset.Class == 'class-3'] = -1

    X_train,Y_train,X_test,Y_test = separate_data(traindataset,testdataset)
    predict(X_train,Y_train,X_test,Y_test)     
    l2norm(X_train,Y_train,X_test,Y_test,traindataset,testdataset)

def class3_all():
    traindataset=pd.read_csv(r'.\train.csv' )
    testdataset=pd.read_csv(r'.\test.csv')
    traindataset.Class[traindataset.Class == 'class-1'] = 1
    traindataset.Class[traindataset.Class == 'class-2'] = 1
    traindataset.Class[traindataset.Class == 'class-3'] = -1
    testdataset.Class[testdataset.Class == 'class-1'] = 1
    testdataset.Class[testdataset.Class == 'class-2'] = 1
    testdataset.Class[testdataset.Class == 'class-3'] = -1

    X_train,Y_train,X_test,Y_test = separate_data(traindataset,testdataset)
    predict(X_train,Y_train,X_test,Y_test) 
    l2norm(X_train,Y_train,X_test,Y_test,traindataset,testdataset)

#this function separates the train and test data into labels and features
def separate_data(traindataset,testdataset):
    X_train=traindataset.iloc[:, :-1].reset_index(drop=True)
    Y_train=traindataset.iloc[:, -1].reset_index(drop=True)#labels
    X_test=testdataset.iloc[:, :-1].reset_index(drop=True)
    Y_test=testdataset.iloc[:, -1].reset_index(drop=True)#labels

    return (X_train,Y_train,X_test,Y_test)     
   
#algorithm for training the perceptron
def train_perceptron(X_train,Y_train):
    w = [0,0,0,0]#weights initialisation to 0
    bias = 0
    activation = 0.0

    for epochs in range(20):#the number of epochs as stated on the assignment
    
        for i in range (len(X_train)):
            activation = X_train.iloc[i] * w #multyplying each value of the row of the data with the corresponding value of weight
            a=activation.sum()+bias#summing up all the values of the row adding bias
            condition = a * Y_train[i] 
            if (condition <= 0 ):
                w += (Y_train[i]*X_train.iloc[i])#weights update
                bias = bias + Y_train[i]#bias update
        
    return w,bias
#this function predicts the correct label    
def predict(X_train,Y_train,X_test,Y_test):
    true_test =0
    false_test = 0
    percent_test = 0
    true_train =0
    false_train = 0
    percent_train = 0
    w,bias = train_perceptron(X_train,Y_train)
    a= 0
    for i in range (len(X_test)):#calculates the test accuracy
        activation = X_test.iloc[i]* w 
        a=activation.sum()+ bias
        
        if (Y_test[i] == 1 ) and (a>0):
            true_test+=1
        elif (Y_test[i] == -1 ) and (a<0):   
            true_test +=1
        else :
            false_test +=1
            
    for k in range (len(Y_train)):#calculates the train accuracy
        activation = X_train.iloc[k]* w 
        a=activation.sum()+ bias         
        if (Y_train[k] == 1 ) and (a>0):
            true_train+=1
        elif (Y_train[k] == -1 ) and (a<0):   
            true_train +=1
        else :
            false_train +=1 
                  
    percent_test= ((true_test) * 100)/ len(X_test)
    print("test data accuracy:", percent_test)
    percent_train= ((true_train) * 100)/ len(X_train)
    print("train data accuracy:",percent_train)
    
#read again the dataset 
def renewdataset(traindataset,testdataset):
    traindataset=pd.read_csv(r'.\train.csv' )
    testdataset=pd.read_csv(r'.\test.csv')
    X_train=traindataset.iloc[:, :-1].reset_index(drop=True)
    X_test=testdataset.iloc[:, :-1].reset_index(drop=True)
    return X_train,X_test
#calculating the l2 regularisation
def l2norm(X_train,Y_train,X_test,Y_test,traindataset,testdataset):
    coefficient=[0.01, 0.1, 1.0, 10.0, 100.0]#the given coefficients

    for k in range(5):
        X_train,X_test= renewdataset(traindataset,testdataset)
        for i in range(len(X_train)):#calculates l2 for train
            l2norm=numpy.sqrt(numpy.sum(X_train.iloc[i] * X_train.iloc[i]))
            X_train.iloc[i] = X_train.iloc[i] / (l2norm*coefficient[k])
            
        for i in range(len(X_test)):#calculates l2 for test
            l2norm=numpy.sqrt(numpy.sum(X_test.iloc[i] * X_test.iloc[i]))
            X_test.iloc[i] = X_test.iloc[i] / (l2norm*coefficient[k]) 
        print("")
        print("L2 regularisation with coefficient", coefficient[k])
        predict(X_train,Y_train,X_test,Y_test)   

print("--- class1-2 ---")
class1_class2()
print("")
print("--- class2-3 ---")
class2_class3()
print("")
print("--- class1-3 ---")
class1_class3() 
print("") 
print("class1 vs rest")
class1_all() 
print("")
print("class2 vs rest")
class2_all()
print("")
print("class3 vs rest")
class3_all()
