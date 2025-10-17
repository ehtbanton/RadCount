from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pathlib import Path
import json
import subprocess
import sys
import platform

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


def home(request):
    """Render the main LLM interface page."""
    llm_service = LlamaService()
    context_files = llm_service.get_context_files_info()

    context = {
        'server_running': llm_service.is_server_running(),
        'model_info': llm_service.get_model_info(),
        'available_models': get_available_models(),
        'current_model': get_current_model(),
        'context_files': context_files,
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