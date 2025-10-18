#!/usr/bin/env python3
"""
Startup script that automatically manages virtual environment and dependencies.
Checks requirements.txt and installs/removes packages as needed before starting Django.
Also manages llama.cpp installation and LLM server startup.
"""
import os
import sys
import subprocess
import platform
import zipfile
import shutil
import time
import signal
import atexit
from pathlib import Path
from urllib.request import urlretrieve

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama_cpp"
MODELS_DIR = PROJECT_ROOT / "models"
CONTEXT_DIR = PROJECT_ROOT / "Context"

# Global variable to track llama.cpp server process
llama_server_process = None

# Platform-specific venv paths
if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"


def print_status(message):
    """Print a status message."""
    print(f"[STARTUP] {message}")


def check_venv_exists():
    """Check if virtual environment exists."""
    return VENV_DIR.exists() and VENV_PYTHON.exists()


def create_venv():
    """Create a new virtual environment."""
    print_status("Virtual environment not found. Creating...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_status("Virtual environment created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create virtual environment: {e}")
        return False


def get_installed_packages():
    """Get a dict of installed packages and their versions."""
    try:
        result = subprocess.run(
            [str(VENV_PIP), "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        packages = json.loads(result.stdout)
        return {pkg["name"].lower(): pkg["version"] for pkg in packages}
    except Exception as e:
        print_status(f"Error getting installed packages: {e}")
        return {}


def get_all_dependencies(packages):
    """Get all dependencies (including transitive) for the given packages."""
    try:
        # Install the packages to ensure we have them and their deps
        result = subprocess.run(
            [str(VENV_PIP), "show"] + packages,
            capture_output=True,
            text=True,
            check=False
        )

        all_deps = set()
        for line in result.stdout.split('\n'):
            if line.startswith('Requires:'):
                deps = line.replace('Requires:', '').strip()
                if deps:
                    for dep in deps.split(','):
                        all_deps.add(dep.strip().lower())

        # Recursively get dependencies of dependencies
        if all_deps:
            nested_deps = get_all_dependencies(list(all_deps))
            all_deps.update(nested_deps)

        return all_deps
    except Exception as e:
        print_status(f"Error getting dependencies: {e}")
        return set()


def get_required_packages():
    """Parse requirements.txt and return required packages."""
    if not REQUIREMENTS_FILE.exists():
        print_status("requirements.txt not found.")
        return {}

    required = {}
    with open(REQUIREMENTS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Simple parsing - handle package names with version specifiers
                package_name = line.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].split("!=")[0].strip()
                required[package_name.lower()] = line
    return required


def sync_dependencies():
    """Synchronize installed packages with requirements.txt."""
    print_status("Checking dependencies...")

    installed = get_installed_packages()
    required = get_required_packages()

    # Filter out pip, setuptools, and wheel from comparison
    base_packages = {"pip", "setuptools", "wheel"}
    installed_filtered = {k: v for k, v in installed.items() if k not in base_packages}

    # Find packages to install (in requirements but not installed or different version)
    to_install = []
    for pkg_name, pkg_spec in required.items():
        if pkg_name not in installed:
            to_install.append(pkg_spec)
            print_status(f"Package '{pkg_name}' needs to be installed.")

    # Install missing packages first
    if to_install:
        print_status(f"Installing {len(to_install)} package(s)...")
        try:
            subprocess.run(
                [str(VENV_PIP), "install"] + to_install,
                check=True
            )
            print_status("Packages installed successfully.")
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to install packages: {e}")
            return False

    # Get all dependencies of required packages (including transitive deps)
    all_required_deps = set(required.keys())
    if required:
        transitive_deps = get_all_dependencies(list(required.keys()))
        all_required_deps.update(transitive_deps)

    # Find packages to remove (installed but not in requirements or their dependencies)
    to_remove = []
    for pkg_name in installed_filtered:
        if pkg_name not in all_required_deps:
            to_remove.append(pkg_name)
            print_status(f"Package '{pkg_name}' is no longer required and will be removed.")

    # Remove extra packages
    if to_remove:
        print_status(f"Removing {len(to_remove)} package(s)...")
        try:
            subprocess.run(
                [str(VENV_PIP), "uninstall", "-y"] + to_remove,
                check=True
            )
            print_status("Packages removed successfully.")
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to remove packages: {e}")
            return False

    if not to_install and not to_remove:
        print_status("All dependencies are up to date.")

    return True


def run_migrations():
    """Run Django migrations if needed."""
    print_status("Checking for pending migrations...")
    manage_py = PROJECT_ROOT / "manage.py"

    if not manage_py.exists():
        print_status("manage.py not found. Skipping migrations.")
        return True

    try:
        # Check if there are unapplied migrations
        result = subprocess.run(
            [str(VENV_PYTHON), str(manage_py), "showmigrations", "--plan"],
            capture_output=True,
            text=True,
            check=True
        )

        # If there are any "[ ]" (unapplied) migrations, run migrate
        if "[ ]" in result.stdout:
            print_status("Applying migrations...")
            subprocess.run(
                [str(VENV_PYTHON), str(manage_py), "migrate"],
                check=True
            )
            print_status("Migrations applied successfully.")
        else:
            print_status("All migrations are up to date.")

        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Migration error: {e}")
        return False


def detect_gpu():
    """Detect available GPU hardware."""
    print_status("Detecting GPU hardware...")

    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            print_status(f"NVIDIA GPU detected: {gpu_name}")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for AMD GPU (basic check)
    if platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            if "AMD" in result.stdout and "VGA" in result.stdout:
                print_status("AMD GPU detected")
                return "rocm"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print_status("Apple Silicon detected")
        return "metal"

    print_status("No GPU detected, using CPU")
    return "cpu"


def check_llama_cpp_installed():
    """Check if llama.cpp is already installed."""
    if not LLAMA_CPP_DIR.exists():
        return False

    # Check for server executable
    if sys.platform == "win32":
        server_exe = LLAMA_CPP_DIR / "llama-server.exe"
    else:
        server_exe = LLAMA_CPP_DIR / "llama-server"

    return server_exe.exists()


def download_llama_cpp(gpu_type):
    """Download and install llama.cpp for the detected GPU type."""
    print_status(f"Downloading llama.cpp ({gpu_type} build)...")

    LLAMA_CPP_DIR.mkdir(exist_ok=True)

    # Determine download URL based on platform and GPU
    # Using pre-built releases from llama.cpp GitHub (latest release: b6736)
    base_url = "https://github.com/ggml-org/llama.cpp/releases/download/b6736"

    system = platform.system()

    if system == "Windows":
        if gpu_type == "cuda":
            filename = "llama-b6736-bin-win-cuda-12.4-x64.zip"
        else:
            filename = "llama-b6736-bin-win-avx2-x64.zip"
    elif system == "Linux":
        if gpu_type == "cuda":
            filename = "llama-b6736-bin-ubuntu-x64-cuda-cu12.4.1.zip"
        else:
            filename = "llama-b6736-bin-ubuntu-x64.zip"
    elif system == "Darwin":
        if platform.machine() == "arm64":
            filename = "llama-b6736-bin-macos-arm64.zip"
        else:
            filename = "llama-b6736-bin-macos-x64.zip"
    else:
        print_status(f"Unsupported platform: {system}")
        return False

    download_url = f"{base_url}/{filename}"
    zip_path = LLAMA_CPP_DIR / filename

    try:
        print_status(f"Downloading from {download_url}...")
        print_status("This may take a few minutes depending on your connection...")
        urlretrieve(download_url, zip_path)

        print_status("Extracting llama.cpp...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(LLAMA_CPP_DIR)

        # Clean up zip file
        zip_path.unlink()

        # Make executables executable on Unix-like systems
        if system != "Windows":
            for exe in LLAMA_CPP_DIR.glob("**/*"):
                if exe.is_file() and not exe.suffix:
                    os.chmod(exe, 0o755)

        print_status("llama.cpp installed successfully")
        return True

    except Exception as e:
        print_status(f"Failed to download llama.cpp: {e}")
        return False


def check_model_downloaded():
    """Check if a model is already downloaded in either the relative models directory or F:/_llm_models/."""
    # Define directories to search
    search_dirs = [MODELS_DIR, Path("F:/_llm_models")]

    for models_dir in search_dirs:
        if not models_dir.exists():
            continue

        # Check for any .gguf file (excluding mmproj files)
        gguf_files = [f for f in models_dir.glob("*.gguf") if not f.name.startswith("mmproj")]
        if gguf_files:
            print_status(f"Found model: {gguf_files[0].name} in {models_dir}")
            return True

    return False


def download_progress_hook(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        if block_num % 50 == 0 or downloaded >= total_size:  # Update every 50 blocks to avoid spam
            # Use carriage return to overwrite the same line
            print(f"\r[STARTUP] Download progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)", end='', flush=True)
            # Print newline when complete
            if downloaded >= total_size:
                print()


def download_vision_model():
    """Download a vision-language model."""
    print_status("Downloading vision-language model (this may take several minutes)...")

    MODELS_DIR.mkdir(exist_ok=True)

    # Try multiple model sources in order of preference
    model_options = [
        # Option 1: LLaVA 1.6 Mistral 7B Q4 (~4GB, recent and capable)
        {
            "url": "https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf",
            "filename": "llava-v1.6-mistral-7b.Q4_K_M.gguf",
            "mmproj_url": "https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf",
            "mmproj_filename": "mmproj-llava-v1.6-f16.gguf"
        },
        # Option 2: LLaVA 1.5 7B Q4 (~4GB, stable)
        {
            "url": "https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf",
            "filename": "llava-v1.5-7b-Q4_K.gguf",
            "mmproj_url": "https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf",
            "mmproj_filename": "mmproj-llava-v1.5-f16.gguf"
        }
    ]

    for i, model_option in enumerate(model_options, 1):
        print_status(f"Trying model option {i}/{len(model_options)}: {model_option['filename']}...")

        model_path = MODELS_DIR / model_option["filename"]
        mmproj_path = MODELS_DIR / model_option.get("mmproj_filename", "")

        try:
            # Download main model
            print_status(f"Downloading {model_option['filename']}...")

            import requests
            headers = {'User-Agent': 'Mozilla/5.0'}

            # Download main model
            response = requests.get(model_option["url"], headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192

            with open(model_path, 'wb') as f:
                for block_num, chunk in enumerate(response.iter_content(chunk_size=block_size)):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        download_progress_hook(block_num, block_size, total_size)

            print_status("Model downloaded successfully")

            # Download mmproj (multimodal projector) if specified
            if "mmproj_url" in model_option and model_option["mmproj_url"]:
                print_status(f"Downloading vision projector {model_option['mmproj_filename']}...")

                response = requests.get(model_option["mmproj_url"], headers=headers, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(mmproj_path, 'wb') as f:
                    for block_num, chunk in enumerate(response.iter_content(chunk_size=block_size)):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            download_progress_hook(block_num, block_size, total_size)

                print_status("Vision projector downloaded successfully")

            return True

        except Exception as e:
            print_status(f"Failed to download model option {i}: {e}")
            # Clean up partial downloads
            if model_path.exists():
                model_path.unlink()
            if mmproj_path and mmproj_path.exists():
                mmproj_path.unlink()

            if i < len(model_options):
                print_status("Trying next model option...")
            else:
                print_status("All model download attempts failed.")
                print_status("You can manually download a vision model (.gguf) to the 'models' folder or F:/_llm_models/")
                print_status("Recommended: LLaVA models from https://huggingface.co/")
                return False

    return False


def get_model_path():
    """Get the path to the downloaded model."""
    if not MODELS_DIR.exists():
        return None

    gguf_files = list(MODELS_DIR.glob("*.gguf"))
    if gguf_files:
        return gguf_files[0]

    return None


def cleanup_llama_server():
    """Cleanup function to stop llama.cpp server on exit."""
    # Note: Server is now managed by the web interface, not by startup.py
    # This function is kept for backward compatibility but does nothing
    pass


def get_mmproj_path():
    """Get the path to the multimodal projector file."""
    if not MODELS_DIR.exists():
        return None

    mmproj_files = list(MODELS_DIR.glob("mmproj*.gguf"))
    if mmproj_files:
        return mmproj_files[0]

    return None


def start_llama_server(gpu_type):
    """Start the llama.cpp server in the background."""
    global llama_server_process

    print_status("Starting llama.cpp server...")

    # Get server executable path
    if sys.platform == "win32":
        server_exe = LLAMA_CPP_DIR / "llama-server.exe"
        # Also check in build/bin for newer versions
        if not server_exe.exists():
            server_exe = LLAMA_CPP_DIR / "build" / "bin" / "llama-server.exe"
        if not server_exe.exists():
            # Check for the files extracted directly
            for potential_exe in LLAMA_CPP_DIR.rglob("llama-server.exe"):
                server_exe = potential_exe
                break
    else:
        server_exe = LLAMA_CPP_DIR / "llama-server"
        if not server_exe.exists():
            server_exe = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"
        if not server_exe.exists():
            for potential_exe in LLAMA_CPP_DIR.rglob("llama-server"):
                server_exe = potential_exe
                break

    if not server_exe.exists():
        print_status(f"llama-server executable not found at {server_exe}")
        print_status("Server will not be started. Please check llama.cpp installation.")
        return False

    model_path = get_model_path()
    if not model_path:
        print_status("No model found. Server cannot start.")
        return False

    # Build command
    cmd = [
        str(server_exe),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", "8080",
        "-c", "4096",  # context size
        "--n-gpu-layers", "33" if gpu_type in ["cuda", "metal"] else "0",
    ]

    # Add mmproj for vision support
    mmproj_path = get_mmproj_path()
    if mmproj_path:
        cmd.extend(["--mmproj", str(mmproj_path)])
        print_status(f"Vision support enabled with {mmproj_path.name}")

    try:
        # Start server as background process
        print_status(f"Starting server with model: {model_path.name}")
        llama_server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(LLAMA_CPP_DIR)
        )

        # Register cleanup
        atexit.register(cleanup_llama_server)

        # Wait a bit to see if it starts successfully
        time.sleep(5)  # Increased wait time for vision models

        if llama_server_process.poll() is None:
            print_status("llama.cpp server started successfully on http://127.0.0.1:8080")
            return True
        else:
            print_status("llama.cpp server failed to start")
            # Try to get error output
            try:
                stderr = llama_server_process.stderr.read().decode('utf-8', errors='ignore')
                if stderr:
                    print_status(f"Server error: {stderr[:500]}")
            except:
                pass
            return False

    except Exception as e:
        print_status(f"Failed to start llama.cpp server: {e}")
        return False


def setup_llm():
    """Setup LLM infrastructure (llama.cpp and model)."""
    print_status("Setting up LLM infrastructure...")

    # Create Context directory if it doesn't exist
    if not CONTEXT_DIR.exists():
        CONTEXT_DIR.mkdir(exist_ok=True)
        print_status("Created Context directory")

    # Detect GPU
    gpu_type = detect_gpu()

    # Check/install llama.cpp
    if not check_llama_cpp_installed():
        print_status("llama.cpp not found, installing...")
        if not download_llama_cpp(gpu_type):
            print_status("Failed to install llama.cpp. Continuing without LLM support.")
            return False
    else:
        print_status("llama.cpp already installed")

    # Check/download model
    if not check_model_downloaded():
        print_status("Model not found, downloading vision-language model...")
        if not download_vision_model():
            print_status("Failed to download model. Continuing without LLM support.")
            return False
    else:
        print_status("Model already downloaded")

    # NOTE: Server is NOT started automatically
    # Users must manually start the server from the web interface
    print_status("LLM infrastructure ready. Start the server from the web interface.")

    return True


def cleanup_llm_server_on_exit():
    """Stop any running LLM server when Django exits."""
    pid_file = PROJECT_ROOT / "llama_server.pid"

    if pid_file.exists():
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())

            print_status(f"Stopping LLM server (PID: {pid})...")

            if sys.platform == "win32":
                subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                             capture_output=True, check=False)
            else:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

            pid_file.unlink()
            print_status("LLM server stopped.")
        except Exception as e:
            print_status(f"Error stopping LLM server: {e}")


def start_django():
    """Start the Django development server."""
    print_status("Starting Django development server...")
    manage_py = PROJECT_ROOT / "manage.py"

    if not manage_py.exists():
        print_status("manage.py not found. Please create a Django project first.")
        return False

    # Register cleanup handler
    atexit.register(cleanup_llm_server_on_exit)

    # Also handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print_status("\nShutting down...")
        cleanup_llm_server_on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Use the venv Python to run Django
        subprocess.run([str(VENV_PYTHON), str(manage_py), "runserver"], check=True)
    except subprocess.CalledProcessError as e:
        print_status(f"Django server error: {e}")
        return False
    except KeyboardInterrupt:
        print_status("Server stopped by user.")
        cleanup_llm_server_on_exit()

    return True


def main():
    """Main startup routine."""
    print_status("Starting RadCount application...")

    # Step 1: Check/create virtual environment
    if not check_venv_exists():
        if not create_venv():
            print_status("Setup failed. Exiting.")
            sys.exit(1)
    else:
        print_status("Virtual environment found.")

    # Step 2: Sync dependencies
    if not sync_dependencies():
        print_status("Dependency synchronization failed. Exiting.")
        sys.exit(1)

    # Step 3: Run migrations
    if not run_migrations():
        print_status("Migration failed. Exiting.")
        sys.exit(1)

    # Step 4: Setup LLM infrastructure (download only, don't start server)
    setup_llm()  # Don't exit on failure, just continue without LLM

    # Step 5: Start Django
    print_status("Setup complete!")
    start_django()


if __name__ == "__main__":
    main()