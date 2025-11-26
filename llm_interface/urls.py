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
    # Ground Truth - Entities and Relations (new entity-first workflow)
    path('add-ground-truth-entity/', views.add_ground_truth_entity, name='add_ground_truth_entity'),
    path('delete-ground-truth-entity/', views.delete_ground_truth_entity, name='delete_ground_truth_entity'),
    path('add-ground-truth-relation/', views.add_ground_truth_relation, name='add_ground_truth_relation'),
    path('delete-ground-truth-relation/', views.delete_ground_truth_relation, name='delete_ground_truth_relation'),
    # Legacy triplet endpoints (for backward compatibility)
    path('add-ground-truth-triplet/', views.add_ground_truth_triplet, name='add_ground_truth_triplet'),
    path('delete-ground-truth-triplet/', views.delete_ground_truth_triplet, name='delete_ground_truth_triplet'),
    # Extraction Methods
    path('create-extraction-method/', views.create_extraction_method, name='create_extraction_method'),
    path('run-extraction-method/', views.run_extraction_method, name='run_extraction_method'),
    path('delete-extraction-method/', views.delete_extraction_method, name='delete_extraction_method'),
    # Two-step LLM extraction
    path('extract-entities-llm/', views.extract_entities_llm, name='extract_entities_llm'),
    path('extract-relations-llm/', views.extract_relations_llm, name='extract_relations_llm'),
    # Metrics
    path('calculate-metrics/', views.calculate_metrics, name='calculate_metrics'),
    # Migration
    path('migrate-entity-ids/', views.migrate_entity_ids, name='migrate_entity_ids'),
    path('schemas/', views.get_schemas, name='get_schemas'),
    path('active-schema/', views.get_active_schema, name='get_active_schema'),
    path('set-active-schema/', views.set_active_schema, name='set_active_schema'),
    path('save-schema/', views.save_schema, name='save_schema'),
    path('delete-schema/', views.delete_schema, name='delete_schema'),
    path('extraction-prompts/', views.get_extraction_prompts_view, name='get_extraction_prompts'),
    path('set-active-prompt/', views.set_active_prompt, name='set_active_prompt'),
    path('save-prompt/', views.save_prompt, name='save_prompt'),
    path('delete-prompt/', views.delete_prompt, name='delete_prompt'),
    # Entity and Relation extraction prompts
    path('entity-extraction-prompts/', views.get_entity_extraction_prompts, name='get_entity_extraction_prompts'),
    path('relation-extraction-prompts/', views.get_relation_extraction_prompts, name='get_relation_extraction_prompts'),
    path('save-entity-extraction-prompt/', views.save_entity_extraction_prompt, name='save_entity_extraction_prompt'),
    path('save-relation-extraction-prompt/', views.save_relation_extraction_prompt, name='save_relation_extraction_prompt'),
    path('execute-extraction/', views.execute_extraction_function, name='execute_extraction_function'),
]