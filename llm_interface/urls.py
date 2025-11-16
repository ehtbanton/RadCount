from django.urls import path
from . import views

app_name = 'llm_interface'

urlpatterns = [
    path('', views.home, name='home'),
    path('generate/', views.generate, name='generate'),
    path('generate-with-prompt/', views.generate_with_prompt, name='generate_with_prompt'),
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
    path('entities/', views.get_entities, name='get_entities'),
    path('save-entities/', views.save_entities, name='save_entities'),
    path('extract-entities-relations/', views.extract_entities_relations, name='extract_entities_relations'),
    path('schemas/', views.get_schemas, name='get_schemas'),
    path('active-schema/', views.get_active_schema, name='get_active_schema'),
    path('set-active-schema/', views.set_active_schema, name='set_active_schema'),
    path('save-schema/', views.save_schema, name='save_schema'),
    path('delete-schema/', views.delete_schema, name='delete_schema'),
    path('extraction-prompts/', views.get_extraction_prompts_view, name='get_extraction_prompts'),
    path('set-active-prompt/', views.set_active_prompt, name='set_active_prompt'),
    path('save-prompt/', views.save_prompt, name='save_prompt'),
    path('delete-prompt/', views.delete_prompt, name='delete_prompt'),
    path('execute-extraction/', views.execute_extraction_function, name='execute_extraction_function'),

    # Ground Truth Management
    path('ground-truth/<int:entry_number>/', views.get_ground_truth, name='get_ground_truth'),
    path('ground-truth/save/', views.save_ground_truth, name='save_ground_truth'),
    path('ground-truth/<int:entry_number>/delete/', views.delete_ground_truth, name='delete_ground_truth'),

    # LLM Extraction Function Management
    path('llm-functions/', views.get_extraction_functions_list, name='get_llm_functions'),
    path('llm-functions/<int:function_id>/', views.get_extraction_function_detail, name='get_llm_function_detail'),
    path('llm-functions/save/', views.save_llm_extraction_function, name='save_llm_function'),
    path('llm-functions/<int:function_id>/delete/', views.delete_llm_extraction_function, name='delete_llm_function'),
    path('llm-functions/<int:function_id>/execute/<int:entry_number>/', views.execute_llm_extraction, name='execute_llm_extraction'),
    path('llm-triplets/<int:entry_number>/<int:function_id>/', views.get_llm_triplets, name='get_llm_triplets'),

    # Evaluation
    path('evaluate/<int:entry_number>/', views.evaluate_entry, name='evaluate_entry'),
    path('evaluate/batch/', views.evaluate_batch, name='evaluate_batch'),
]