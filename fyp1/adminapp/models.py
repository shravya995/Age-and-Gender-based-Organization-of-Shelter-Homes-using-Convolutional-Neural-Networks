
from django.db import models

import os


# Create your models here.
GENDER_CHOICES = (
	('female','FEMALE'),
	('male','MALE'),
)

"""def content_file_name(instance, filename):

	ext = filename.split('.')[-1]
	filename = "%s.%s" % ("img", ext)
	return os.path.join('images',filename)"""

def rename(instance,filename):
	extension = filename.split('.')[-1]
	filename ="%s.%s" % (instance.name,extension)
	return os.path.join('images',filename)
"""class AdminForm(models.Model):

	name = models.CharField(max_length=200)
	created_at = models.DateTimeField(auto_now_add=True)
	age = models.IntegerField(blank=True)
	gender = models.CharField(max_length=200, choices=GENDER_CHOICES,blank=True)
	address = models.CharField(max_length=200)
	description = models.CharField(max_length=600,blank=True)
	image = models.ImageField(upload_to=content_file_name)
	

   
	def __str__(self):
		return self.name"""

class AdminadmitForm(models.Model):
	name = models.CharField(max_length=200)
	created_at = models.DateTimeField(auto_now_add=True)
	age = models.IntegerField(blank=True)
	gender = models.CharField(max_length=200, choices=GENDER_CHOICES,blank=True)
	address = models.CharField(max_length=200)
	description = models.CharField(max_length=600,blank=True)
	image = models.ImageField(upload_to=rename)
	

   
	def __str__(self):
		return self.name
class SparshaMan(models.Model):
	age = models.IntegerField()
	gender = models.CharField(max_length=200)
	name= models.CharField(max_length=200)
	def __str__(self):

		return self.age
	def __str__(self):
		return self.gender

class SparshaWoman(models.Model):
	age = models.IntegerField()
	gender = models.CharField(max_length=200)
	name = models.CharField(max_length=200)
	def __str__(self):

		return self.age
	def __str__(self):
		return self.gender

class ShishuMandir(models.Model):
	age = models.IntegerField()
	gender = models.CharField(max_length=200)
	name= models.CharField(max_length=200)
	def __str__(self):

		return self.age
	def __str__(self):
		return self.gender

class AdarshaDala(models.Model):
	age = models.IntegerField()
	gender = models.CharField(max_length=200)
	name= models.CharField(max_length=200)
	def __str__(self):

		return self.age
	def __str__(self):
		return self.gender

class HasiruDala(models.Model):
	age = models.IntegerField()
	gender = models.CharField(max_length=200)
	name= models.CharField(max_length=200)
	def __str__(self):

		return self.age
	def __str__(self):
		return self.gender

class ShantiSayog(models.Model):
	age = models.IntegerField()
	gender = models.CharField(max_length=200)
	name= models.CharField(max_length=200)
	def __str__(self):

		return self.age
	def __str__(self):
		return self.gender
