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
    # Annotations (simplified observation-location pairs)
    path('add-annotation/', views.add_annotation, name='add_annotation'),
    path('delete-annotation/', views.delete_annotation, name='delete_annotation'),
    # Simplified LLM extraction
    path('extract-observations/', views.extract_observations, name='extract_observations'),
    # Binary classification extraction
    path('extract-binary-classification/', views.extract_binary_classification, name='extract_binary_classification'),
    path('get-binary-classification-prompt/', views.get_binary_classification_prompt, name='get_binary_classification_prompt'),
    path('save-binary-classification-prompt/', views.save_binary_classification_prompt_view, name='save_binary_classification_prompt'),
    # Extraction Methods
    path('create-extraction-method/', views.create_extraction_method, name='create_extraction_method'),
    path('run-extraction-method/', views.run_extraction_method, name='run_extraction_method'),
    path('delete-extraction-method/', views.delete_extraction_method, name='delete_extraction_method'),
    # Two-step LLM extraction
    path('extract-entities-llm/', views.extract_entities_llm, name='extract_entities_llm'),
    path('extract-relations-llm/', views.extract_relations_llm, name='extract_relations_llm'),
    # One-shot extraction
    path('extract-oneshot/', views.extract_oneshot, name='extract_oneshot'),
    # Metrics
    path('calculate-metrics/', views.calculate_metrics, name='calculate_metrics'),
    # Migration
    path('migrate-entity-ids/', views.migrate_entity_ids, name='migrate_entity_ids'),
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
    # Test & Evaluation
    path('calculate-observation-metrics/', views.calculate_observation_metrics, name='calculate_observation_metrics'),
    path('save-test-results/', views.save_test_results, name='save_test_results'),
    path('get-test-versions/', views.get_test_versions, name='get_test_versions'),
    path('get-current-version-info/', views.get_current_version_info, name='get_current_version_info'),
    # Findings extraction
    path('extract-findings/', views.extract_findings, name='extract_findings'),
    path('calculate-findings-metrics/', views.calculate_findings_metrics, name='calculate_findings_metrics'),
    path('get-findings-prompt/', views.get_findings_prompt, name='get_findings_prompt'),
    path('save-findings-prompt/', views.save_findings_prompt, name='save_findings_prompt'),
    path('get-triplet-prompt/', views.get_triplet_prompt, name='get_triplet_prompt'),
    path('save-triplet-prompt/', views.save_triplet_prompt, name='save_triplet_prompt'),
    # Schema management (for Entities & Relations tab)
    path('schemas/', views.get_schemas, name='get_schemas'),
    path('active-schema/', views.get_active_schema, name='get_active_schema'),
    path('set-active-schema/', views.set_active_schema, name='set_active_schema'),
    path('save-schema/', views.save_schema, name='save_schema'),
    path('delete-schema/', views.delete_schema, name='delete_schema'),
    # Ground Truth - Entities and Relations (new entity-first workflow)
    path('add-ground-truth-entity/', views.add_ground_truth_entity, name='add_ground_truth_entity'),
    path('delete-ground-truth-entity/', views.delete_ground_truth_entity, name='delete_ground_truth_entity'),
    path('add-ground-truth-relation/', views.add_ground_truth_relation, name='add_ground_truth_relation'),
    path('delete-ground-truth-relation/', views.delete_ground_truth_relation, name='delete_ground_truth_relation'),
    # Experiment Logging
    path('experiments/', views.experiments, name='experiments'),
    path('experiments/<str:experiment_id>/', views.experiment_detail, name='experiment_detail'),
    path('experiments-compare/', views.compare_experiments, name='compare_experiments'),
    path('experiments-export/', views.export_experiments, name='export_experiments'),
    path('experiment-types/', views.get_experiment_types, name='get_experiment_types'),
]