from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from django import forms 
from .models import UseradmitForm,Item_donation


class CreateUserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['first_name','last_name','username','email','password1','password2']

class userForm(ModelForm):
   
    class Meta:
        model = UseradmitForm
        fields = ['name', 'age', 'gender', 'address','description','image']
        labels = {
            'name' : '',
            'age' :  '',
            'gender' : '',
            'address' :  '',
            'description' :  '',
            'image' : ''
        }
        widgets = {
            'name' : forms.TextInput(attrs={'class':'form-control','placeholder':'Name'}),
            'age' :  forms.TextInput(attrs={'class':'form-control','placeholder':'Age'}),
            'gender' :  forms.TextInput(attrs={'class':'form-control','placeholder':'Gender'}),
            'address' :  forms.TextInput(attrs={'class':'form-control','placeholder':'Address'}),
            'description' :  forms.Textarea(attrs={'class':'form-control','placeholder':'Description'}),
            'image' :  forms.FileInput(attrs={'class':'form-control'})
                   }

class Itemdonation(ModelForm):
   
    class Meta:
        model = Item_donation
        fields = ['Full_name', 'phoneno', 'address', 'shelterhomes','items']
        labels = {
            'Full_name' : '',
            'phoneno' :  '',
            'address' : '',
            'shelterhomes' :  '',
            'items' :  '',
            
        }
        widgets = {
            'Full_name' : forms.TextInput(attrs={'class':'form-control','placeholder':'Please enter Full name'}),
            'phoneno' :  forms.TextInput(attrs={'class':'form-control','placeholder':'Phone Number'}),
            'address' :  forms.TextInput(attrs={'class':'form-control','placeholder':'Address'}),
            'shelterhomes' :  forms.Select(attrs={'class':'form-control','placeholder':'Please select you shelter home','choices':'SHELTER_CHOICES'}),
            'items' :  forms.Select(attrs={'class':'form-control','placeholder':'items','choices':'ITEMS_CHOICES'}),
            
                   }
