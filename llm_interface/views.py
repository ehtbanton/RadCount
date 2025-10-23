from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pathlib import Path
import json
import subprocess
import sys
import platform
import csv

from .llm_service import LlamaService

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama_cpp"


def detect_gpu():
    """Detect available GPU hardware."""
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"

    return "cpu"


def get_available_models():
    """Get list of available models from both the models directory and F:/_llm_models/."""
    models = []

    # Define directories to search
    search_dirs = [MODELS_DIR]
    external_models_dir = Path("F:/_llm_models")
    if external_models_dir.exists():
        search_dirs.append(external_models_dir)

    # Search all directories
    for models_dir in search_dirs:
        if not models_dir.exists():
            continue

        for model_file in models_dir.glob("*.gguf"):
            # Skip mmproj files
            if not model_file.name.startswith("mmproj"):
                models.append({
                    'filename': model_file.name,
                    'path': str(model_file),
                    'size_mb': model_file.stat().st_size / 1024 / 1024
                })

    return models


def get_current_model():
    """Get the currently loaded model filename."""
    model_file = PROJECT_ROOT / "llama_server.model"
    if model_file.exists():
        try:
            with open(model_file, 'r') as f:
                return f.read().strip()
        except:
            pass
    return None


def set_current_model(model_filename):
    """Store the currently loaded model filename."""
    model_file = PROJECT_ROOT / "llama_server.model"
    with open(model_file, 'w') as f:
        f.write(model_filename)


def clear_current_model():
    """Clear the currently loaded model."""
    model_file = PROJECT_ROOT / "llama_server.model"
    if model_file.exists():
        model_file.unlink()


def get_large_files():
    """Get list of large data files from large_data directory."""
    large_files = []
    large_data_dir = PROJECT_ROOT / "large_data"

    if not large_data_dir.exists():
        return large_files

    for file_path in large_data_dir.iterdir():
        if file_path.is_file():
            large_files.append({
                'filename': file_path.name,
                'size_mb': file_path.stat().st_size / 1024 / 1024
            })

    return large_files


def get_csv_metadata():
    """Get metadata about the CSV file (if it exists)."""
    large_data_dir = PROJECT_ROOT / "large_data"

    if not large_data_dir.exists():
        return None

    # Look for a CSV file
    csv_files = list(large_data_dir.glob("*.csv"))

    if not csv_files:
        return None

    csv_file = csv_files[0]

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read first row as headers

            # Count rows (excluding header)
            row_count = sum(1 for _ in reader)

        return {
            'filename': csv_file.name,
            'headers': headers,
            'row_count': row_count,
            'size_mb': csv_file.stat().st_size / 1024 / 1024
        }
    except Exception as e:
        return None


def home(request):
    """Render the main LLM interface page."""
    llm_service = LlamaService()
    context_files = llm_service.get_context_files_info()

    # Calculate total file count
    total_files = (len(context_files['system_files']) +
                   len(context_files['user_files']) +
                   len(context_files['image_files']))

    # Check if we're editing a file
    edit_filename = request.GET.get('edit')
    edit_content = None
    if edit_filename:
        # Security check
        if '..' not in edit_filename and '/' not in edit_filename and '\\' not in edit_filename:
            if edit_filename.endswith('.txt'):
                context_dir = PROJECT_ROOT / "Context"
                file_path = context_dir / edit_filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            edit_content = f.read()
                    except:
                        pass

    # Get large files information
    large_files = get_large_files()
    csv_metadata = get_csv_metadata()

    context = {
        'server_running': llm_service.is_server_running(),
        'model_info': llm_service.get_model_info(),
        'available_models': get_available_models(),
        'current_model': get_current_model(),
        'context_files': context_files,
        'total_context_files': total_files,
        'large_files': large_files,
        'total_large_files': len(large_files),
        'csv_metadata': csv_metadata,
        'edit_filename': edit_filename,
        'edit_content': edit_content,
    }

    return render(request, 'llm_interface/home.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def generate(request):
    """Generate a response from the LLM based on context files."""
    try:
        # Parse request parameters
        data = json.loads(request.body) if request.body else {}
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 512))

        # Generate response
        llm_service = LlamaService()

        if not llm_service.is_server_running():
            return JsonResponse({
                'success': False,
                'error': 'LLM server is not running. Please check startup logs.'
            }, status=503)

        result = llm_service.generate_response(
            temperature=temperature,
            max_tokens=max_tokens
        )

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': f'Invalid parameter value: {str(e)}'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def status(request):
    """Get the status of the LLM server and model info."""
    llm_service = LlamaService()

    return JsonResponse({
        'server_running': llm_service.is_server_running(),
        'model_info': llm_service.get_model_info(),
        'available_models': get_available_models(),
        'current_model': get_current_model(),
    })


@require_http_methods(["GET"])
def token_count(request):
    """Get the total token count for all context files."""
    llm_service = LlamaService()

    if not llm_service.is_server_running():
        return JsonResponse({
            'success': False,
            'error': 'LLM server is not running. Start the server to calculate tokens.'
        }, status=503)

    try:
        counts = llm_service.get_total_context_token_count()
        return JsonResponse({
            'success': True,
            'counts': counts
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to calculate token count: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def start_server(request):
    """Start the llama.cpp server with specified model."""
    try:
        data = json.loads(request.body) if request.body else {}
        model_filename = data.get('model')

        if not model_filename:
            return JsonResponse({
                'success': False,
                'error': 'No model specified'
            }, status=400)

        # Find the model file in either directory
        model_path = MODELS_DIR / model_filename
        if not model_path.exists():
            # Try external models directory
            external_model_path = Path("F:/_llm_models") / model_filename
            if external_model_path.exists():
                model_path = external_model_path
            else:
                return JsonResponse({
                    'success': False,
                    'error': f'Model file not found: {model_filename}'
                }, status=404)

        # Check if server is already running
        llm_service = LlamaService()
        if llm_service.is_server_running():
            return JsonResponse({
                'success': False,
                'error': 'Server is already running. Stop it first before starting with a different model.'
            }, status=400)

        # Find llama-server executable
        if sys.platform == "win32":
            server_exe = None
            for potential_exe in LLAMA_CPP_DIR.rglob("llama-server.exe"):
                server_exe = potential_exe
                break
            if not server_exe:
                server_exe = LLAMA_CPP_DIR / "llama-server.exe"
        else:
            server_exe = None
            for potential_exe in LLAMA_CPP_DIR.rglob("llama-server"):
                server_exe = potential_exe
                break
            if not server_exe:
                server_exe = LLAMA_CPP_DIR / "llama-server"

        if not server_exe or not server_exe.exists():
            return JsonResponse({
                'success': False,
                'error': 'llama-server executable not found. Please run startup.py to install llama.cpp.'
            }, status=500)

        # Check for mmproj file
        mmproj_path = None
        for mmproj_file in MODELS_DIR.glob("mmproj*.gguf"):
            mmproj_path = mmproj_file
            break

        # Detect GPU for optimal settings
        gpu_type = detect_gpu()
        gpu_layers = "33" if gpu_type in ["cuda", "metal"] else "0"

        # Build command
        cmd = [
            str(server_exe),
            "-m", str(model_path),
            "--host", "127.0.0.1",
            "--port", "8080",
            "-c", "4096",
            "--n-gpu-layers", gpu_layers,
        ]

        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])

        # Start server
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(LLAMA_CPP_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )

        # Store process ID for later stopping
        import os
        pid_file = PROJECT_ROOT / "llama_server.pid"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

        # Store the current model
        set_current_model(model_filename)

        return JsonResponse({
            'success': True,
            'message': f'Server starting with model: {model_filename}',
            'pid': process.pid
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to start server: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def stop_server(request):
    """Stop the llama.cpp server."""
    try:
        pid_file = PROJECT_ROOT / "llama_server.pid"

        if not pid_file.exists():
            return JsonResponse({
                'success': False,
                'error': 'No server PID file found. Server may not be running.'
            }, status=404)

        # Read PID
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        # Kill process
        if sys.platform == "win32":
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=False)
        else:
            import signal
            import os
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        # Remove PID file
        pid_file.unlink()

        # Clear current model
        clear_current_model()

        return JsonResponse({
            'success': True,
            'message': 'Server stopped successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to stop server: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def upload_context(request):
    """Upload a context file to the Context directory."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No file provided'
            }, status=400)

        uploaded_file = request.FILES['file']
        filename = uploaded_file.name

        # Validate file extension
        allowed_extensions = ['.txt', '.jpg', '.jpeg', '.png']
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return JsonResponse({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            }, status=400)

        # Ensure Context directory exists
        context_dir = PROJECT_ROOT / "Context"
        context_dir.mkdir(exist_ok=True)

        # Save the file
        file_path = context_dir / filename
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        return JsonResponse({
            'success': True,
            'message': f'File {filename} uploaded successfully',
            'filename': filename
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to upload file: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_context(request):
    """Get the content of a context file."""
    try:
        filename = request.GET.get('filename')

        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        # Only allow .txt files for editing
        if not filename.endswith('.txt'):
            return JsonResponse({
                'success': False,
                'error': 'Only text files can be edited'
            }, status=400)

        context_dir = PROJECT_ROOT / "Context"
        file_path = context_dir / filename

        if not file_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'File {filename} not found'
            }, status=404)

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return JsonResponse({
            'success': True,
            'filename': filename,
            'content': content
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to read file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def save_context(request):
    """Save or update a context file (always overwrites if exists)."""
    try:
        data = json.loads(request.body) if request.body else {}
        filename = data.get('filename', '').strip()
        content = data.get('content', '')

        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)

        # Validate filename
        if not filename.endswith('.txt'):
            return JsonResponse({
                'success': False,
                'error': 'Filename must end with .txt'
            }, status=400)

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        # Ensure Context directory exists
        context_dir = PROJECT_ROOT / "Context"
        context_dir.mkdir(exist_ok=True)

        file_path = context_dir / filename

        # Save the file (overwrite if exists)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return JsonResponse({
            'success': True,
            'message': f'File {filename} saved successfully',
            'filename': filename
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to save file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_context(request):
    """Delete a context file from the Context directory."""
    try:
        data = json.loads(request.body) if request.body else {}
        filename = data.get('filename')

        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        context_dir = PROJECT_ROOT / "Context"
        file_path = context_dir / filename

        if not file_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'File {filename} not found'
            }, status=404)

        # Delete the file
        file_path.unlink()

        return JsonResponse({
            'success': True,
            'message': f'File {filename} deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def upload_large(request):
    """Upload a CSV file to the large_data directory. Only one CSV file is allowed at a time."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No file provided'
            }, status=400)

        uploaded_file = request.FILES['file']
        filename = uploaded_file.name

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        # Validate that it's a CSV file
        if not filename.lower().endswith('.csv'):
            return JsonResponse({
                'success': False,
                'error': 'Only CSV files are allowed'
            }, status=400)

        # Ensure large_data directory exists
        large_data_dir = PROJECT_ROOT / "large_data"
        large_data_dir.mkdir(exist_ok=True)

        # Delete any existing CSV files (only one CSV file allowed)
        for existing_csv in large_data_dir.glob("*.csv"):
            existing_csv.unlink()

        # Save the file
        file_path = large_data_dir / filename
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Validate CSV format
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Try to read headers
                if not headers:
                    file_path.unlink()  # Delete invalid file
                    return JsonResponse({
                        'success': False,
                        'error': 'CSV file is empty or invalid'
                    }, status=400)
        except Exception as e:
            file_path.unlink()  # Delete invalid file
            return JsonResponse({
                'success': False,
                'error': f'Invalid CSV format: {str(e)}'
            }, status=400)

        return JsonResponse({
            'success': True,
            'message': f'CSV file {filename} uploaded successfully',
            'filename': filename
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to upload file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_large(request):
    """Delete a large file from the large_data directory."""
    try:
        data = json.loads(request.body) if request.body else {}
        filename = data.get('filename')

        if not filename:
            return JsonResponse({
                'success': False,
                'error': 'No filename provided'
            }, status=400)

        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return JsonResponse({
                'success': False,
                'error': 'Invalid filename'
            }, status=400)

        large_data_dir = PROJECT_ROOT / "large_data"
        file_path = large_data_dir / filename

        if not file_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'File {filename} not found'
            }, status=404)

        # Delete the file
        file_path.unlink()

        return JsonResponse({
            'success': True,
            'message': f'File {filename} deleted successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete file: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_csv_entry(request):
    """Get a specific entry from the CSV file by row number (1-indexed, excluding header)."""
    try:
        entry_num = request.GET.get('entry')

        if not entry_num:
            return JsonResponse({
                'success': False,
                'error': 'No entry number provided'
            }, status=400)

        try:
            entry_num = int(entry_num)
        except ValueError:
            return JsonResponse({
                'success': False,
                'error': 'Entry number must be an integer'
            }, status=400)

        if entry_num < 1:
            return JsonResponse({
                'success': False,
                'error': 'Entry number must be at least 1'
            }, status=400)

        # Find the CSV file
        large_data_dir = PROJECT_ROOT / "large_data"
        if not large_data_dir.exists():
            return JsonResponse({
                'success': False,
                'error': 'No CSV file found'
            }, status=404)

        csv_files = list(large_data_dir.glob("*.csv"))
        if not csv_files:
            return JsonResponse({
                'success': False,
                'error': 'No CSV file found'
            }, status=404)

        csv_file = csv_files[0]

        # Read the CSV file
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read headers

            # Skip to the desired row
            for i, row in enumerate(reader, start=1):
                if i == entry_num:
                    # Create a dictionary mapping headers to values
                    entry_data = {}
                    for j, header in enumerate(headers):
                        if j < len(row):
                            entry_data[header] = row[j]
                        else:
                            entry_data[header] = ""

                    return JsonResponse({
                        'success': True,
                        'entry_number': entry_num,
                        'data': entry_data
                    })

            # If we got here, the entry number is out of range
            return JsonResponse({
                'success': False,
                'error': f'Entry {entry_num} not found. File has {i} entries.'
            }, status=404)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to read CSV entry: {str(e)}'
        }, status=500)