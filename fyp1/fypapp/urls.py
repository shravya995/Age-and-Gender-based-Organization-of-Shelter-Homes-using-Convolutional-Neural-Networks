from django.urls import path
from fypapp import views

urlpatterns = [
    path('',views.index, name="index"),
    path('register/',views.registerPage, name='registerPage'),
    path('login/',views.loginPage, name='loginPage'),
    path('logout/',views.logoutUser,name='logout'),
    path('home/',views.home,name='home'),
    path('useraddPerson',views.useraddPerson,name='useraddPerson',),
    path('shelterhomes',views.shelterhomes,name='shelterhomes'),
    path('donite/',views.dona,name='donite'),
    path('payment',views.payment,name='payment'),
    path('success',views.success,name='success')
]