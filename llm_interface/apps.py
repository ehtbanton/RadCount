from django.apps import AppConfig
import os


class LlmInterfaceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llm_interface'

    def ready(self):
        # Only auto-start in the main process (not the reloader)
        if os.environ.get('RUN_MAIN') == 'true':
            self._auto_start_llm_server()

    def _auto_start_llm_server(self):
        from .views import (
            get_available_models, get_current_model, get_current_context_size,
            start_llm_server_process,
        )
        from .llm_service import LlamaService

        llm_service = LlamaService()
        if llm_service.is_server_running():
            return

        models = get_available_models()
        if not models:
            return

        model_filename = get_current_model() or models[0]['filename']
        context_size = get_current_context_size()

        # Verify the chosen model still exists on disk
        model_filenames = {m['filename'] for m in models}
        if model_filename not in model_filenames:
            model_filename = models[0]['filename']

        result = start_llm_server_process(model_filename, context_size)
        if result['success']:
            print(f"[LLM] Auto-started server with model: {model_filename}")
        else:
            print(f"[LLM] Auto-start failed: {result.get('error', 'unknown error')}")
