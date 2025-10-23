from django.urls import path
from . import views

app_name = 'llm_interface'

urlpatterns = [
    path('', views.home, name='home'),
    path('generate/', views.generate, name='generate'),
    path('status/', views.status, name='status'),
    path('token-count/', views.token_count, name='token_count'),
    path('start/', views.start_server, name='start_server'),
    path('stop/', views.stop_server, name='stop_server'),
    path('upload/', views.upload_context, name='upload_context'),
    path('save/', views.save_context, name='save_context'),
    path('delete/', views.delete_context, name='delete_context'),
    path('upload-large/', views.upload_large, name='upload_large'),
    path('delete-large/', views.delete_large, name='delete_large'),
    path('csv-entry/', views.get_csv_entry, name='get_csv_entry'),
    path('save-csv-entry/', views.save_csv_entry_to_context, name='save_csv_entry_to_context'),
    path('extraction-functions/', views.get_extraction_functions, name='get_extraction_functions'),
    path('save-extraction-function/', views.save_extraction_function, name='save_extraction_function'),
    path('delete-extraction-function/', views.delete_extraction_function, name='delete_extraction_function'),
    path('execute-extraction/', views.execute_extraction_function, name='execute_extraction_function'),
]