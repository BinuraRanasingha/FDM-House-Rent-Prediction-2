#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', 'C:\\Users\\Binura\\Desktop\\Project_11_1_2022\\House_Rent_Prediction\\dataset')
#important include the directory to the cloned folder


# In[2]:


import pandas as pd
train_df = pd.read_csv("train_reg.csv")
X_train=pd.read_csv("Xtrain_classify.csv")
Y_train=pd.read_csv("Ytrain_classify.csv")


# In[3]:


get_ipython().run_line_magic('cd', 'C:\\Users\\Binura\\Desktop\\Project_11_1_2022\\House_Rent_Prediction')
#important include the directory to the cloned folder


# In[4]:


import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

app = Flask(__name__)
model = pickle.load(open('model/update_reg3.0.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/size')
def size():
    return render_template('reg.html')

@app.route('/est')
def est():
    return render_template('clas.html')


@app.route('/clasify',methods=['POST'])
def clasify():
    '''
    For rendering results on HTML GUI for Classification
    
    '''
    
    arr=np.zeros(18) #Intializing array of zero
    
    bath1=request.form.get("bath")
    room1=request.form.get("room")
    size1=request.form.get("size")
    area1= request.form.get('area')
    tenant= request.form.get('ten')
    furn= request.form.get('furn')
    city= request.form.get('city')
   
    
    #Assigning values for the array
    arr[0]=room1
    arr[1]=bath1
    arr[2]=size1
    
    if "sup" in area1:
        arr[5]="1"
    elif "carp" in area1:
        arr[4]="1"
    elif "buil" in area1:
        arr[3]="1"
    
    if "b" in tenant:
        arr[6]="1"
    elif "f" in tenant:
        arr[8]="1"
    elif "both" in tenant:
        arr[7]="1"
    
    
    if "f" in furn:
        arr[9]="1"
    elif "sf" in furn:
        arr[10]="1"
    elif "none" in furn:
        arr[11]="1"
        
    if "bangalore" in city:
        arr[12]="1"
    elif "chennai" in city:
        arr[13]="1"
    elif "delhi" in city:
        arr[14]="1"
    elif "hyd" in city:
        arr[15]="1"
    elif "kol" in city:
        arr[16]="1"
    elif "mum" in city:
        arr[17]="1"
 
        
        
    #converting to 2D array
    X_test=arr.reshape(1,18)
    #create KNN classifier
    KNN_classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  

    #train the model
    KNN_classifier.fit(X_train, Y_train.to_numpy())  

    #predict the output using the model
    y_pred=KNN_classifier.predict(X_test)
    
    
    def convert(y_pred):
        result=""
        if y_pred == 0:
            result="1500-13350"
        elif y_pred ==1:
            result="13350-25200"
        elif y_pred ==2:
            result="25200-37050"
        elif y_pred ==3:
            result="37050-48900"
        elif y_pred ==4:
            result="48900-60750"
        elif y_pred ==5:
            result="60750-72600"
        elif y_pred ==6:
            result="72600-84450"
        elif y_pred ==7:
            result="84450-96300"
        elif y_pred ==8: 
            result="96300-108150"
        elif y_pred ==9: 
            result="108150-120000"
        else:
            result="No class"
        
        return result
        
    

    #output = round(prediction[0], 2)
  

    return render_template('clas.html', prediction_text=convert(y_pred))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI for Regression
    
    '''
    bath=request.form.get("bath")
    room=request.form.get("room")
    size=request.form.get("size")
    select = request.form.get('area')
   
    data=[bath,room,size,select]
    
    pdata=np.array(data).reshape(1,4)
    scaler = StandardScaler().fit(train_df)
    rescaled_X_test = scaler.transform(pdata)
    prediction = model.predict(rescaled_X_test)
    actual_predicted = np.exp(prediction)

    #output = round(prediction[0], 2)
    output=data

    return render_template('reg.html', prediction_text=actual_predicted)



if __name__ == "__main__":
    print(__name__)
    app.run(debug=False)
    


# In[5]:


arr=np.zeros(18)
arr


# In[6]:


len(arr)

