from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('test/', views.predict, name='test'),
    path('consult_doctor/', views.consult_doctor, name='consult_doctor'),
]
