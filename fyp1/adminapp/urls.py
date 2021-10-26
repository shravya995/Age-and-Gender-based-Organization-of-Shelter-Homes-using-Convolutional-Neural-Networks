from django.urls import path
from adminapp import views
urlpatterns = [
  
    
    path('adminlogin/',views.adminloginPage, name='adminloginPage'),
    path('logout/',views.logoutAdmin,name='logout'),
    path('adminhome/',views.adminhome,name='adminhome'),
    path('addPerson/',views.addPerson,name='addPerson'),
    path('modelResult/',views.run_model,name='modelResult'),
    path('getimage/',views.get_image,name='getimage'),
    path('removedata/',views.data_remove,name='removedata'),
    
   
    
]