from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser
import os 

# Create your models here.
GENDER_CHOICES = (
    ('female','FEMALE'),
    ('male','MALE'),
)
SHELTER_CHOICES = [
    ('Please select shelter home','Please select shelter home'),
    ('Shishu Mandir','Shishu Mandir'),
    ('Adarsha Dala','Adarsha Dala'),
    ('Hasiru Dala','Hasiru Dala'),
    ('Sparsha Trust(Womens)','Sparsha Trust(Womens)'),
    ('Sparsha Trust(Mens)','Sparsha Trust(Mens)'),
    ('Shanti Sayog','Shanti Sayog'),
]

ITEMS_CHOICES = [
    ('Please select Items you wish to donate','Please select Items you wish to donate'),
    ('CLOTHING','CLOTHING'),
    ('BOOKS','BOOKS'),
    ('FURNITURE','FURNITURE'),
    ('MEDICINES','MEDICINES'),
    ('GROCERY KIT','GROCERY KIT'),
    ('ELECTRONIC GADGET','ELECTRONIC GADGET')
]
def content_file_name(instance, filename):

    ext = filename.split('.')[-1]
    filename = "%s.%s" % (instance.name, ext)
    return os.path.join('images', filename)
"""class UserForm(models.Model):
    User= settings.AUTH_USER_MODEL
    user= models.ForeignKey(User,on_delete=models.DO_NOTHING)
    submitted_by= models.CharField(max_length=50)
    email_id=models.EmailField(max_length=200)
    created_at = models.DateField(auto_now_add=True)
    name = models.CharField(max_length=200)
    age = models.IntegerField(blank=True)
    gender = models.CharField(max_length=200, choices=GENDER_CHOICES,blank=True)
    address = models.CharField(max_length=200)
    description = models.CharField(max_length=600,blank=True)
    image = models.ImageField(upload_to=content_file_name)
   

    def __str__(self):
        return self.name"""

class UseradmitForm(models.Model):
    User= settings.AUTH_USER_MODEL
    user= models.ForeignKey(User,on_delete=models.DO_NOTHING)
    submitted_by= models.CharField(max_length=50)
    email_id=models.EmailField(max_length=200)
    created_at = models.DateField(auto_now_add=True)
    name = models.CharField(max_length=200)
    age = models.IntegerField(blank=True)
    gender = models.CharField(max_length=200, choices=GENDER_CHOICES,blank=True)
    address = models.CharField(max_length=200)
    description = models.CharField(max_length=600,blank=True)
    image = models.ImageField(upload_to=content_file_name)

    def __str__(self):
        return self.name

class Item_donation(models.Model):
    User= settings.AUTH_USER_MODEL
    user= models.ForeignKey(User,on_delete=models.DO_NOTHING)
    submitted_by= models.CharField(max_length=50)
    created_at = models.DateField(auto_now_add=True)
    Full_name = models.CharField(max_length=200)
    phoneno = models.CharField(max_length=12)
    address = models.CharField(max_length=200)
    shelterhomes = models.CharField(max_length=200,choices=SHELTER_CHOICES,blank=False, default=1)
    items = models.CharField(max_length=200,choices=ITEMS_CHOICES,blank=False, default=1)

    def __str__(self):
        return self.name

