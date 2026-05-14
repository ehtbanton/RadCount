#!/usr/bin/env python3
"""
RadCount startup script — fully automated setup and launch.
Handles: venv, GPU-aware PyTorch, pip dependencies, llama.cpp,
LLM model download (with resume), Django migrations, and server launch.

Can be run directly (python startup.py) or via setup.bat which
also bootstraps Python if it's not installed.
"""
import os
import sys
import re
import subprocess
import platform
import zipfile
import shutil
import signal
import atexit
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ─── Constants ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama_cpp"
MODELS_DIR = PROJECT_ROOT / "llm_models"
CONTEXT_DIR = PROJECT_ROOT / "Context"

if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"

LLAMA_CPP_VERSION = "b6736"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    "/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
DEFAULT_MODEL_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MIN_PYTHON = (3, 10)  # Django 5.0 requires 3.10+
MIN_VRAM_WARNING_MB = 6000
REQUIRED_DISK_GB = 12


# ─── Output Helpers ───────────────────────────────────────────

def print_status(msg):
    print(f"  [*] {msg}")

def print_success(msg):
    print(f"  [+] {msg}")

def print_warning(msg):
    print(f"  [!] {msg}")

def print_error(msg):
    print(f"  [-] {msg}")

def print_header(msg):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


# ─── Pre-flight Checks ───────────────────────────────────────

def check_python_version():
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        print_error(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required "
            f"(found {v.major}.{v.minor}.{v.micro})"
        )
        print_error("Install from https://www.python.org/downloads/")
        return False
    print_success(f"Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_venv_module():
    try:
        import venv  # noqa: F401
        return True
    except ImportError:
        print_error("Python 'venv' module not available.")
        if platform.system() == "Linux":
            print_error("  Debian/Ubuntu: sudo apt install python3-venv")
            print_error("  Fedora:        sudo dnf install python3-venv")
        return False


def detect_gpu():
    """Detect GPU type, name, VRAM, and max supported CUDA version."""
    info = {"type": "cpu", "name": None, "vram_mb": 0, "cuda_version": None}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split("\n")[0].split(",")]
            info["name"] = parts[0] if parts else "Unknown NVIDIA GPU"
            info["vram_mb"] = int(parts[1]) if len(parts) > 1 else 0
            info["type"] = "cuda"

            # Parse max CUDA version from nvidia-smi banner
            header = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10,
            )
            if header.returncode == 0:
                m = re.search(r"CUDA Version:\s*(\d+\.\d+)", header.stdout)
                if m:
                    info["cuda_version"] = m.group(1)
            return info
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info.update(type="metal", name="Apple Silicon")
        return info

    return info


def check_disk_space():
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
    if free_gb < REQUIRED_DISK_GB:
        print_warning(
            f"Low disk space: {free_gb:.1f} GB free "
            f"({REQUIRED_DISK_GB} GB recommended for all downloads)"
        )
    else:
        print_success(f"Disk space: {free_gb:.1f} GB free")


# ─── Virtual Environment ─────────────────────────────────────

def ensure_venv():
    if VENV_DIR.exists() and VENV_PYTHON.exists():
        print_success("Virtual environment found")
        return True

    print_status("Creating virtual environment...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)], check=True,
        )
        print_success("Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def upgrade_pip():
    try:
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True, check=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass  # non-critical


# ─── Dependencies ─────────────────────────────────────────────

def _check_torch_cuda():
    """Return (is_installed, has_cuda, version_string)."""
    try:
        r = subprocess.run(
            [str(VENV_PYTHON), "-c",
             "import torch; print(torch.cuda.is_available()); print(torch.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()
            return True, lines[0].strip() == "True", lines[1].strip() if len(lines) > 1 else "?"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return False, False, None


def _pytorch_index_url(gpu_info):
    """Pick the right PyTorch wheel index for the detected CUDA version."""
    if gpu_info["type"] != "cuda":
        return None  # default PyPI → CPU torch

    cuda = gpu_info.get("cuda_version") or ""
    if not cuda:
        return "https://download.pytorch.org/whl/cu124"

    major, minor = (int(x) for x in cuda.split(".")[:2])
    if major >= 12 and minor >= 4:
        return "https://download.pytorch.org/whl/cu124"
    if major >= 12:
        return "https://download.pytorch.org/whl/cu121"
    if major == 11 and minor >= 8:
        return "https://download.pytorch.org/whl/cu118"

    print_warning(f"CUDA {cuda} is too old for GPU-accelerated PyTorch")
    return None


def install_torch(gpu_info):
    installed, has_cuda, version = _check_torch_cuda()

    if installed and has_cuda:
        print_success(f"PyTorch {version} (CUDA) already installed")
        return True
    if installed and gpu_info["type"] != "cuda":
        print_success(f"PyTorch {version} (CPU) already installed")
        return True

    index_url = _pytorch_index_url(gpu_info)

    # If CPU torch is present but we want CUDA, remove the CPU build first
    # so pip doesn't consider the requirement satisfied.
    if installed and not has_cuda and index_url:
        print_status("Replacing CPU PyTorch with CUDA build...")
        subprocess.run(
            [str(VENV_PIP), "uninstall", "-y", "torch"],
            capture_output=True, check=False,
        )

    cmd = [str(VENV_PIP), "install", "torch"]
    if index_url:
        cmd += ["--index-url", index_url]
        print_status("Installing PyTorch with CUDA support (may take a few minutes)...")
    else:
        print_status("Installing PyTorch...")

    try:
        subprocess.run(cmd, check=True, timeout=600)
        _, got_cuda, ver = _check_torch_cuda()
        tag = " (CUDA)" if got_cuda else " (CPU)"
        print_success(f"PyTorch {ver}{tag} installed")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if index_url:
            print_warning("CUDA install failed — falling back to CPU PyTorch...")
            try:
                subprocess.run(
                    [str(VENV_PIP), "install", "torch"],
                    check=True, timeout=600,
                )
                print_success("PyTorch (CPU fallback) installed")
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        print_error("Failed to install PyTorch")
        return False


def install_requirements():
    if not REQUIREMENTS_FILE.exists():
        print_warning("requirements.txt not found")
        return True

    print_status("Installing remaining dependencies...")
    try:
        subprocess.run(
            [str(VENV_PIP), "install", "-r", str(REQUIREMENTS_FILE)],
            check=True, timeout=600,
        )
        print_success("Dependencies installed")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print_error(f"Dependency install failed: {e}")
        return False


# ─── Django Migrations ────────────────────────────────────────

def run_migrations():
    manage_py = PROJECT_ROOT / "manage.py"
    if not manage_py.exists():
        print_warning("manage.py not found — skipping migrations")
        return True
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(manage_py), "showmigrations", "--plan"],
            capture_output=True, text=True, check=True,
        )
        if "[ ]" in result.stdout:
            print_status("Applying database migrations...")
            subprocess.run(
                [str(VENV_PYTHON), str(manage_py), "migrate"], check=True,
            )
            print_success("Migrations applied")
        else:
            print_success("Database up to date")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Migration error: {e}")
        return False


# ─── Downloads ────────────────────────────────────────────────

def download_file(url, dest_path, label=""):
    """Download a URL to dest_path with progress and resume support.

    Writes to a .partial temp file, then renames on completion so
    interrupted downloads can resume on the next run.
    """
    partial = dest_path.parent / (dest_path.name + ".partial")

    resume_from = 0
    if partial.exists():
        resume_from = partial.stat().st_size
        print_status(f"Resuming from {resume_from / 1024 / 1024:.0f} MB...")

    headers = {"User-Agent": "Mozilla/5.0 (RadCount-Setup)"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    try:
        resp = urlopen(Request(url, headers=headers), timeout=30)
    except HTTPError as e:
        if e.code == 416 and resume_from > 0:
            # Already fully downloaded
            if dest_path.exists():
                dest_path.unlink()
            partial.rename(dest_path)
            return True
        raise

    cl = resp.headers.get("Content-Length")
    total = int(cl) + resume_from if cl else 0

    downloaded = resume_from
    last_pct = -10

    with open(partial, "ab" if resume_from else "wb") as f:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                if pct - last_pct >= 5 or downloaded >= total:
                    mb = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    print(
                        f"\r  [*] {label}: {pct}% "
                        f"({mb:.0f}/{mb_total:.0f} MB)   ",
                        end="", flush=True,
                    )
                    last_pct = pct

    if total > 0:
        print()  # newline after progress bar

    if dest_path.exists():
        dest_path.unlink()
    partial.rename(dest_path)
    return True


# ─── LLM Infrastructure ──────────────────────────────────────

def check_llama_cpp_installed():
    if not LLAMA_CPP_DIR.exists():
        return False
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return any(True for _ in LLAMA_CPP_DIR.rglob(exe))


def _llama_cpp_url(gpu_info):
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_VERSION}"
    ver = LLAMA_CPP_VERSION
    system = platform.system()

    if system == "Windows":
        if gpu_info["type"] == "cuda":
            return f"{base}/llama-{ver}-bin-win-cuda-12.4-x64.zip"
        return f"{base}/llama-{ver}-bin-win-avx2-x64.zip"
    if system == "Linux":
        if gpu_info["type"] == "cuda":
            return f"{base}/llama-{ver}-bin-ubuntu-x64-cuda-cu12.4.1.zip"
        return f"{base}/llama-{ver}-bin-ubuntu-x64.zip"
    if system == "Darwin":
        arch = "arm64" if platform.machine() == "arm64" else "x64"
        return f"{base}/llama-{ver}-bin-macos-{arch}.zip"
    return None


def download_llama_cpp(gpu_info):
    url = _llama_cpp_url(gpu_info)
    if not url:
        print_error(f"No llama.cpp build for {platform.system()}")
        return False

    LLAMA_CPP_DIR.mkdir(exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    zip_path = LLAMA_CPP_DIR / filename

    tag = "CUDA" if gpu_info["type"] == "cuda" else "CPU"
    print_status(f"Downloading llama.cpp ({tag})...")

    try:
        download_file(url, zip_path, f"llama.cpp {tag}")
        print_status("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(LLAMA_CPP_DIR)
        zip_path.unlink()

        if platform.system() != "Windows":
            for f in LLAMA_CPP_DIR.rglob("*"):
                if f.is_file() and not f.suffix:
                    os.chmod(f, 0o755)

        print_success("llama.cpp installed")
        return True
    except Exception as e:
        print_error(f"llama.cpp download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def check_model_downloaded():
    if not MODELS_DIR.exists():
        return False
    gguf = [f for f in MODELS_DIR.glob("*.gguf") if not f.name.startswith("mmproj")]
    if gguf:
        print_success(f"Model: {gguf[0].name}")
        return True
    return False


def download_model():
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / DEFAULT_MODEL_FILENAME

    print_status(f"Downloading {DEFAULT_MODEL_FILENAME} (~4.9 GB)")
    print_status("This will take several minutes...")

    try:
        download_file(DEFAULT_MODEL_URL, dest, "Model")
        print_success("Model downloaded")
        return True
    except Exception as e:
        print_error(f"Model download failed: {e}")
        print_status("You can manually place a .gguf model in the 'llm_models' folder")
        print_status("Re-run this script to resume a partial download")
        return False


def setup_llm(gpu_info):
    CONTEXT_DIR.mkdir(exist_ok=True)

    if not check_llama_cpp_installed():
        if not download_llama_cpp(gpu_info):
            return False
    else:
        print_success("llama.cpp already installed")

    if not check_model_downloaded():
        if not download_model():
            return False

    return True


# ─── Django Server ────────────────────────────────────────────

def cleanup_llm_server_on_exit():
    pid_file = PROJECT_ROOT / "llama_server.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
        print_status(f"Stopping LLM server (PID {pid})...")
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, check=False,
            )
        else:
            os.kill(pid, signal.SIGTERM)
        pid_file.unlink()
    except (ProcessLookupError, ValueError, OSError):
        pass


def start_django():
    manage_py = PROJECT_ROOT / "manage.py"
    if not manage_py.exists():
        print_error("manage.py not found")
        return False

    atexit.register(cleanup_llm_server_on_exit)

    def on_signal(sig, frame):
        print("\n")
        print_status("Shutting down...")
        cleanup_llm_server_on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, on_signal)

    try:
        print_success("Starting Django on http://127.0.0.1:8000")
        print_status("LLM server will auto-start when Django loads (port 8080)")
        print_status("Press Ctrl+C to stop\n")
        subprocess.run([str(VENV_PYTHON), str(manage_py), "runserver"], check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"Django error: {e}")
        return False
    except KeyboardInterrupt:
        cleanup_llm_server_on_exit()

    return True


# ─── Main ─────────────────────────────────────────────────────

def main():
    print_header("RadCount Startup")

    # ── Pre-flight ──
    print_header("Pre-flight Checks")

    if not check_python_version():
        sys.exit(1)
    if not check_venv_module():
        sys.exit(1)

    gpu_info = detect_gpu()
    if gpu_info["type"] == "cuda":
        vram = f"{gpu_info['vram_mb']} MB" if gpu_info["vram_mb"] else "unknown"
        cuda = f"CUDA {gpu_info['cuda_version']}" if gpu_info["cuda_version"] else "CUDA"
        print_success(f"GPU: {gpu_info['name']} ({vram} VRAM, {cuda})")
        if gpu_info["vram_mb"] and gpu_info["vram_mb"] < MIN_VRAM_WARNING_MB:
            print_warning(f"Low VRAM ({gpu_info['vram_mb']} MB). 8 GB+ recommended.")
            print_warning("The LLM may run slowly or fail to load the default model.")
    elif gpu_info["type"] == "metal":
        print_success(f"GPU: {gpu_info['name']}")
    else:
        print_warning("No GPU detected — LLM will run on CPU (much slower).")

    check_disk_space()

    # ── Virtual environment ──
    print_header("Virtual Environment")

    if not ensure_venv():
        sys.exit(1)
    upgrade_pip()

    # ── Dependencies ──
    print_header("Dependencies")

    install_torch(gpu_info)
    if not install_requirements():
        sys.exit(1)

    # ── Database ──
    print_header("Database")

    if not run_migrations():
        sys.exit(1)

    # ── LLM infrastructure ──
    print_header("LLM Infrastructure")

    llm_ok = setup_llm(gpu_info)
    if not llm_ok:
        print_warning("LLM setup incomplete — web UI will work but LLM features won't.")
        print_warning("Re-run this script to retry downloads.")

    # ── Launch ──
    print_header("Ready")
    start_django()


if __name__ == "__main__":
    main()
