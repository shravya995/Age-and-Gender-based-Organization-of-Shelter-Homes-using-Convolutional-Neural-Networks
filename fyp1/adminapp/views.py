from django.shortcuts import render

# Create your views here.
from django.shortcuts import render,redirect
from django.contrib.auth import authenticate , login, logout 
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse
from PIL import Image, ImageOps
from django.conf import settings
from django.core.mail import send_mail

import os
import smtplib
import imghdr
from email.message import EmailMessage

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras import layers


from .decorators import allowed_users
from .forms import adminForm
from .models import AdminadmitForm,SparshaMan,SparshaWoman,HasiruDala,AdarshaDala,ShishuMandir,ShantiSayog
from fypapp.models import UseradmitForm
# Create your views here.
def adminloginPage(request):

	if request.method == 'POST':
		username=request.POST.get('username')
		password=request.POST.get('password')

		user = authenticate(request, username=username ,password=password)
		if user is not None:
			login(request ,user)
			return redirect('adminhome')
		else:
			messages.info(request,'Username OR password is incorrect')
		  
	context = {}
	return render(request,'admin_login.html',context)
def logoutAdmin(request):
	logout(request)
	return redirect('adminloginPage')

"""@login_required(login_url='adminloginPage')"""
@allowed_users(allowed_roles=['admin'])
def adminhome(request):
	
	all_forms = UseradmitForm.objects.all()
	context = {'all_forms': all_forms}
	return render(request,'adminhome.html',context)

"""@login_required(login_url='adminloginPage')"""

"""def addPerson(request):
	submitted = False
   
	if request.method == 'POST':
	   
		form = adminForm(request.POST,request.FILES)
		if form.is_valid():
			form.save()
			return HttpResponseRedirect('/addPerson?submitted=True')
	else:
		form = adminForm
		if 'submitted' in request.GET:
			submitted = True
	
		ad_id=AdminadmitForm.objects.latest('created_at')
		
		new_name=ad_id.name
		img_path=ad_id.image
		
		#add_id=ad_id.admission_id
		print(new_name)

		print(img_path)
	   
		
		age,gender=run_model(new_name)
		print(age)
		print(gender)
		category=" "

		if age<15:
			
			category="Shishu Mandir Orphanage"
			ShishuMandir.objects.create(age=age,gender=gender,name=new_name)
		elif age>50:
			
			category="Shanti Sayog Olg Age Home"
			ShantiSayog.objects.create(age=age,gender=gender,name=new_name)

		elif age>15 and age<18:
			if gender==0:
				category="Adarsha Dala Men(youth) Shelter Home"
				AdarshaDala.objects.create(age=age,gender=gender,name=new_name)
			else:
				category="Hasiru Dala Women(youth) Shelter Home"
				HasiruDala.objects.create(age=age,gender=gender,name=new_name)

		else:
			if gender==0:
				category="Sparsha Trust Men Shelter Home"
				SparshaMan.objects.create(age=age,gender=gender,name=new_name)
			   
			else:
				category="Sparsha Trust Women Shelter Home"
				SparshaWoman.objects.create(age=age,gender=gender,name=new_name)
	
	print(category)
	return render(request, 'admin_form.html',{'form':form,'submitted':submitted,'category':category})"""

def addPerson(request):
	submitted = False
	category=""
   
	if request.method == 'POST':
	   
		form = adminForm(request.POST,request.FILES)
		if form.is_valid():
			form.save()
			return HttpResponseRedirect('/addPerson?submitted=True')
			
	else:
		form = adminForm
		if 'submitted' in request.GET:
			submitted = True
			ad_id=AdminadmitForm.objects.latest('created_at')
		
			new_name=ad_id.name
			img_path=ad_id.image
		
			#add_id=ad_id.admission_id
			print(new_name)

			print(img_path)
	   
		
			age,gender=run_model(new_name)
			print(age)
			print(gender)
			category=" "

			if age<15:
				category="Shishu Mandir Orphanage"
				ShishuMandir.objects.create(age=age,gender=gender,name=new_name)
			elif age>44:
				category="Shanti Sayog Olg Age Home"
				ShantiSayog.objects.create(age=age,gender=gender,name=new_name)
			elif age>15 and age<18:
				if gender==0:
					category="Adarsha Dala Men(youth) Shelter Home"
					AdarshaDala.objects.create(age=age,gender=gender,name=new_name)
				else:
					category="Hasiru Dala Women(youth) Shelter Home"
					HasiruDala.objects.create(age=age,gender=gender,name=new_name)
			else:
				if gender==0:
					category="Sparsha Trust Men Shelter Home"
					SparshaMan.objects.create(age=age,gender=gender,name=new_name)
					
				else:
					category="Sparsha Trust Women Shelter Home"
					SparshaWoman.objects.create(age=age,gender=gender,name=new_name)
			
			print(category)
	

	
	return render(request, 'admin_form.html',{'form':form,'submitted':submitted,'category':category})

def Convolution(input_tensor,filters):
	
	x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.001))(input_tensor)
	x = Dropout(0.1)(x)
	x= Activation('relu')(x)
	return x

def get_model():

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

def run_model(img_name):
	"""filename=os.listdir('/Users/snehavenkatesh/Desktop/fyp1/media/images')[-1]
	print(type(filename))
	print(filename)
	print(type(img_name))"""
	#cv2.imwrite('/Users/snehavenkatesh/Desktop/fyp/media/temp.jpg',img_name)
	model=get_model()
	model.load_weights('/Users/snehavenkatesh/Desktop/age&gender/weights.61-6.81.hdf5')
	path=(f'/Users/snehavenkatesh/Desktop/fyp1/media/images/{img_name}.jpg')
	"""image_name='img_'+str(new_name)+'.jpg'"""
	"""face = cv2.imread('/Users/snehavenkatesh/Desktop/fyp/media/images/'img_'+str().jpg',cv2.IMREAD_COLOR)"""
	path_name=path
	try:

		ImageOps.expand(Image.open(path_name),border=25,fill='white').save(path_name)
		print(path_name)
		p=25
		img = cv2.imread(path_name)
		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
		(x, y, w, h) = faces_detected[0]
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
		cv2.imwrite(path_name, img[y-p+1:y+h+p, x-p+1:x+w+p])
		face = cv2.imread(path_name,cv2.IMREAD_COLOR)
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face =cv2.resize(face, (48,48) )
		
		face = np.expand_dims(face,axis=0)
		print(face.shape)
		print(model.predict([[face]]))
		results=model.predict([[face]])
		
	except:
		face = cv2.imread(path_name,cv2.IMREAD_COLOR)
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face =cv2.resize(face, (48,48) )
		
		face = np.expand_dims(face,axis=0)
		print(face.shape)
		print(model.predict([[face]]))
		results=model.predict([[face]])

	age=0
	gender=0
	if(results[0]<=0.44):
		gender=0
	else:
		gender=1
	age=results[1][0][0]
   
	
	return age,gender
	
	
def get_image(request):
	
	#ad_id= request.GET.get('a_id')
	name=request.GET.get('nam')
	print(name)
	print(name.split('?'))
	new_name,email=name.split('?')
	print(new_name)
	print(email)
	

	age,gender=run_model(new_name)
	category=0
	if age<15:
		print(1)
		category="Shishu Mandir Orphanage"
		ShishuMandir.objects.create(age=age,gender=gender,name=new_name)
	elif age>44:
		print(2)
		category="Shanti Sayog Olg Age Home"
		ShantiSayog.objects.create(age=age,gender=gender,name=new_name)

	elif age>15 and age<18:
		if gender==0:
			category="Adarsha Dala Men(youth) Shelter Home"
			AdarshaDala.objects.create(age=age,gender=gender,name=new_name)

		else:
			category="Hasiru Dala Women(youth) Shelter Home"
			HasiruDala.objects.create(age=age,gender=gender,name=new_name)
	else:
		if gender==0:
			category="Sparsha Trust Men Shelter Home"
			SparshaMan.objects.create(age=age,gender=gender,name=new_name)
		   
		else:
			category="Sparsha Trust Women Shelter Home"
			SparshaWoman.objects.create(age=age,gender=gender,name=new_name)
	
	subject = 'shelter home allocation'
	message = f'Hi , the person {new_name} is allocated to {category}'
	email_from = settings.EMAIL_HOST_USER
	
	recipient_list = [email]
   
	
	send_mail( subject, message, email_from, recipient_list )
	
	UseradmitForm.objects.filter(name=new_name).delete()
	

	return render(request,'shelterhome_alloc.html',{'new_name':new_name,'category':category})

def data_remove(request):
	new_name=request.GET.get('nam')
	UseradmitForm.objects.filter(name=new_name).delete()
	#data.delete()
	#return render(request,'adminhome.html')
	return redirect('adminhome')




