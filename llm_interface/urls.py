from django.urls import path
from . import views

app_name = 'llm_interface'

urlpatterns = [
    path('', views.home, name='home'),
    path('generate/', views.generate, name='generate'),
    path('status/', views.status, name='status'),
    path('start/', views.start_server, name='start_server'),
    path('stop/', views.stop_server, name='stop_server'),
]