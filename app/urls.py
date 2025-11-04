# simulator/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),                 # Home
    path('cpu_scheduler/', views.cpu, name='cpu_scheduler'),
    path('pager/', views.paging, name='pager'),
 path('sjf/', views.sjf, name='sjf'),
    path("chatbot_api/", views.chatbot_api, name="chatbot_api"),
  
   
    
]   
