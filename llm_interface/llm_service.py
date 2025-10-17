"""
LLM Service for interacting with llama.cpp server.
"""
import os
import base64
import requests
from pathlib import Path
from typing import List, Dict, Optional


class LlamaService:
    """Service class to interact with llama.cpp server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url
        self.context_dir = Path(__file__).parent.parent / "Context"

    def is_server_running(self) -> bool:
        """Check if llama.cpp server is running."""
        try:
            # Try /health endpoint first
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            pass

        # If /health doesn't work, try /v1/models (alternative endpoint)
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            pass

        # Try a simple connection test
        try:
            response = requests.get(f"{self.base_url}/", timeout=2)
            return response.status_code in [200, 404]  # 404 is OK, means server is up
        except requests.exceptions.RequestException:
            return False

    def get_model_info(self) -> Optional[Dict]:
        """Get information about the currently loaded model."""
        try:
            response = requests.get(f"{self.base_url}/props", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None

    def get_context_files_info(self) -> Dict[str, List]:
        """
        Get information about context files without loading full content.
        Returns dict with lists of file info for display purposes.
        """
        context = {
            'system_files': [],
            'user_files': [],
            'image_files': []
        }

        if not self.context_dir.exists():
            return context

        for file_path in self.context_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                file_size = file_path.stat().st_size

                # Handle text files
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    preview = content[:200] + '...' if len(content) > 200 else content

                    file_info = {
                        'filename': filename,
                        'size': file_size,
                        'content': content,
                        'preview': preview
                    }

                    if filename.startswith('sys-'):
                        context['system_files'].append(file_info)
                    else:
                        context['user_files'].append(file_info)

                # Handle image files
                elif filename.endswith(('.jpg', '.jpeg', '.png')):
                    context['image_files'].append({
                        'filename': filename,
                        'size': file_size,
                        'path': str(file_path)
                    })

        return context

    def load_context_files(self) -> Dict[str, List]:
        """
        Load all context files from Context folder.
        Returns dict with 'system_prompts', 'user_prompts', and 'images'.
        """
        context = {
            'system_prompts': [],
            'user_prompts': [],
            'images': []
        }

        if not self.context_dir.exists():
            return context

        for file_path in self.context_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name

                # Handle text files
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    if filename.startswith('sys-'):
                        context['system_prompts'].append(content)
                    else:
                        context['user_prompts'].append(content)

                # Handle image files
                elif filename.endswith(('.jpg', '.jpeg', '.png')):
                    with open(file_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    context['images'].append({
                        'filename': filename,
                        'data': image_data
                    })

        return context

    def generate_response(self, temperature: float = 0.7, max_tokens: int = 512) -> Dict:
        """
        Generate a response based on context files.

        Args:
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dict with 'success', 'response', and optional 'error' keys
        """
        try:
            # Load context
            context = self.load_context_files()

            # Build prompt
            messages = []

            # Add system prompts
            if context['system_prompts']:
                system_content = '\n\n'.join(context['system_prompts'])
                messages.append({
                    'role': 'system',
                    'content': system_content
                })

            # Build user message with text and images
            user_content = []

            # Add text prompts
            if context['user_prompts']:
                user_text = '\n\n'.join(context['user_prompts'])
                user_content.append({
                    'type': 'text',
                    'text': user_text
                })

            # Add images
            for image in context['images']:
                user_content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{image['data']}"
                    }
                })

            if user_content:
                messages.append({
                    'role': 'user',
                    'content': user_content if len(user_content) > 1 else user_content[0]['text']
                })

            # Make request to llama.cpp server
            payload = {
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'stream': False
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content']
                return {
                    'success': True,
                    'response': generated_text,
                    'context_files': {
                        'system_prompts': len(context['system_prompts']),
                        'user_prompts': len(context['user_prompts']),
                        'images': len(context['images'])
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f"Server returned status {response.status_code}: {response.text}"
                }

        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timed out. The model might be too slow or the prompt too long.'
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Request failed: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }