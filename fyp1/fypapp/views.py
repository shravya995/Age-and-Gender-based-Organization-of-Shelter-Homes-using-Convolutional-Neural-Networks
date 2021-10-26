
from django.shortcuts import render,redirect
from . import forms
from django.contrib.auth.forms import UserCreationForm
from .forms import CreateUserForm
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.contrib.auth import authenticate , login, logout 
from django.contrib.auth.decorators import login_required
from .decorators import allowed_users
from .forms import userForm,Itemdonation
from .models import UseradmitForm,Item_donation

import razorpay
from django.views.decorators.csrf import csrf_exempt


"""import numpy as np
import pandas as pd 
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers
# Create your views here."""

def index(request):
    return render(request,'index.html')
def loginPage(request):

    if request.method == 'POST':
        username=request.POST.get('username')
        password=request.POST.get('password')

        user = authenticate(request, username=username ,password=password)
        if user is not None:
            login(request ,user)
            return redirect('home')
        else:
            messages.info(request,'Username OR password is incorrect')
          
    context = {}
    return render(request,'user_login.html',context)


def logoutUser(request):
    logout(request)
    return redirect('loginPage')
def registerPage(request):
    form= CreateUserForm()

    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(request, 'Account was created for ' + user)
            return redirect('loginPage')
    context = {'form':form}
    return render(request, 'user_register.html',context)

@login_required(login_url='login')
def home(request):
    return render(request,'home.html')

@login_required(login_url='login')
def useraddPerson(request):
    submitted = False
    
    if request.method == 'POST':
       
        form = userForm(request.POST,request.FILES)
        if form.is_valid():
            fs=form.save(commit=False)
            fs.user=request.user
            fs.submitted_by=request.user.username
            fs.email_id=request.user.email
            fs.save()
            return HttpResponseRedirect('/useraddPerson?submitted=True')
    else:
        form = userForm
        if 'submitted' in request.GET:
            submitted = True
     
    return render(request, 'user_form.html',{'form':form,'submitted':submitted})

@login_required(login_url='login')
def shelterhomes(request):
    return render(request,'shelterhomes.html')



@login_required(login_url='login')
def dona(request):
    submitted = False
    
    if request.method == 'POST':
       
        form = Itemdonation(request.POST,request.FILES)
        if form.is_valid():
            fs=form.save(commit=False)
            fs.user=request.user
            fs.submitted_by=request.user.username
            fs.save()
            return HttpResponseRedirect('/donite?submitted=True')
    else:
        form = Itemdonation
        if 'submitted' in request.GET:
            submitted = True
     
    return render(request, 'donaitem.html',{'form':form,'submitted':submitted})

@login_required(login_url='login')
def payment(request):
    if request.method == "POST":
        
        amount = 50000
        client = razorpay.Client(auth=("rzp_test_OcQr0PNW7OfdiB", "4FTeWMIf1ZXcs4TwqUkNmClD"))
        payment = client.order.create({'amount': amount, 'currency': 'INR','payment_capture': '1'})
    return render(request, 'payment.html')

@csrf_exempt
def success(request):
    return render(request, "success.html")