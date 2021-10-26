from django import forms
from django.forms import ModelForm
from .models import AdminadmitForm

class adminForm(ModelForm):
   
    class Meta:
        model = AdminadmitForm
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
      