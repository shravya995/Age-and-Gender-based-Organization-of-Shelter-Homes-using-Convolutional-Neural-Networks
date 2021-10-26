#!/usr/bin/env python
# coding: utf-8

# ## UTK

# In[1]:


import numpy as np
import pandas as pd 
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
print(os.listdir("D:/datasets/UTKface_Aligned_cropped/"))


# In[2]:


def imshow(img):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([]) 
    plt.show()


# In[3]:


onlyfiles = os.listdir("D:/datasets/UTKface_Aligned_cropped/UTKFace")
y_utk = np.array([[[i.split('_')[0]],[i.split('_')[1]]] for i in onlyfiles])
# y = np.array([[i.split('_')[1] for i in onlyfiles]]).T
print(y_utk.shape)
print(y_utk[0])


# In[4]:


X_data =[]
i=0
for file in onlyfiles:
    print(i)
    i+=1
    face = cv2.imread("D:/datasets/UTKface_Aligned_cropped/UTKFace/"+file,cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face =cv2.resize(face, (48,48) )
    X_data.append(face)
X_data=np.array(X_data)
X_data.shape


# In[5]:


X_UTK = np.squeeze(X_data)


# ## FAIRFACE

# In[6]:


data=pd.DataFrame(pd.read_csv(r"D:/datasets/fairface/FairFace/train_labels.csv"))


# In[7]:


print(len(data))
#86744


# In[8]:


data = data[(data.race == "Indian")]


# In[9]:


len(data)
#12319


# In[10]:


data.head()


# In[11]:


print(os.listdir("D:/datasets/fairface/FairFace/"))


# In[12]:


gender=[]
for gen in data['gender']:
    if gen=='Male':
        gender.append(str(0))
    elif gen=='Female':
        gender.append(str(1))


# In[13]:


print(len(gender))


# In[14]:


def avg_age(age1,age2):
    return((age1+age2)/2)


# In[15]:


age=[]
for a in data['age']:
    if a=='more than 70':
        age.append(str(80))
    else:
        ages=a.split('-')
        age.append(str(avg_age(int(ages[0]),int(ages[1]))))


# In[16]:


for i in range(5):
    print(age[i],gender[i])


# In[17]:


y=[]
for i in range(len(age)):
    unit=[[age[i]],[gender[i]]]
    y.append(unit)


# In[18]:


y_fairface=np.array(y)


# In[19]:


print(y_fairface.shape)
y_fairface


# In[20]:


onlyfiles = data['file']


# In[21]:


X_data =[]
for file in onlyfiles:
    print(file)
    face = cv2.imread("D:/datasets/fairface/FairFace/"+file,cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face =cv2.resize(face, (48,48) )
    X_data.append(face)
X_data=np.array(X_data)
X_data.shape


# In[22]:


print(X_data[0])


# In[23]:


X_fairface= np.squeeze(X_data)


# ## Comparision

# In[24]:


len(X_UTK)


# In[25]:


len(X_fairface)


# In[26]:


len(y_utk)


# In[27]:


len(y_fairface)


# In[28]:


y_utk.shape


# In[29]:


y_fairface.shape


# In[30]:


X=X_UTK


# In[31]:


X=np.concatenate((X_UTK,X_fairface))


# In[32]:


len(X)


# In[33]:


y=np.concatenate((y_utk,y_fairface))


# In[34]:


len(y)


# In[35]:


n=30001
imshow(X[n])
print(y[n])


# ## Train_test_split

# In[36]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)
y_train=[y_train[:,1],y_train[:,0]]
y_valid=[y_valid[:,1],y_valid[:,0]]


# In[37]:


from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras import layers
import tensorflow as tf

def gen_model():
    inputs = Input(shape=(48, 48, 3))
    x = inputs
    x = layers.Conv2D(52,3,activation='relu')(x)
    x = layers.Conv2D(52,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(84,3,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x1 = layers.Dense(64,activation='relu')(x)
    x2 = layers.Dense(64,activation='relu')(x)
    x1 = layers.Dense(1,activation='sigmoid',name='sex_out')(x1)
    x2 = layers.Dense(1,activation='relu',name='age_out')(x2)
    model = tf.keras.models.Model(inputs=inputs, outputs=[x1, x2])
    model.compile(optimizer='Adam', loss=['binary_crossentropy','mae'],metrics=['accuracy']) 
    return model
model=gen_model()


# In[38]:


model.summary()


# In[41]:


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:


import random
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
random_id=random.random()
model.summary()
output_path='D:/projects/test'
callbacks = [ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]

history=model.fit(X_train, y_train, epochs=60,batch_size=240,validation_data=(X_valid,y_valid),shuffle=True)


# In[46]:





# In[51]:


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['sex_out_acc'])
plt.plot(history.history['val_sex_out_acc'])
plt.title('sex model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['sex_out_acc'])
plt.plot(history.history['val_sex_out_acc'])
plt.title('sex model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[46]:


#load model directly
model.load_weights('D:/projects/Age_and_Gender_fairFace_utk/weights.61-6.81.hdf5')


# In[47]:


model.evaluate(X_valid,y_valid)


# In[70]:


from PIL import Image, ImageOps
path='C:/Users/Shravya/Downloads/test/test_img_10.jpg'
try:
    ImageOps.expand(path,border=25,fill='white').save('imaged-with-border.jpg')
    p=25
    img = cv2.imread('D:/projects/imaged-with-border.jpg')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    (x, y, w, h) = faces_detected[0]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1);
    imshow(img)


    cv2.imwrite('crop.jpg', img[y-p+1:y+h+p, x-p+1:x+w+p])
    face = cv2.imread('D:\projects\crop.jpg',cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face =cv2.resize(face, (48,48) )
    imshow(face)
    results=model.predict([[face]])
    print(results)
except:
    face = cv2.imread(path,cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face =cv2.resize(face, (48,48) )
    imshow(face)
    results=model.predict([[face]])
    print(results)


# In[35]:


age=0
gender=0
if(results[0]<=0.5):
    gender=0
else:
    gender=1
age=results[1][0][0]


# In[31]:


print("age:",age,"gender:",gender)


# In[ ]:




