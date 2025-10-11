#!/usr/bin/env python3
"""
Startup script that automatically manages virtual environment and dependencies.
Checks requirements.txt and installs/removes packages as needed before starting Django.
"""
import os
import sys
import subprocess
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

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


def start_django():
    """Start the Django development server."""
    print_status("Starting Django development server...")
    manage_py = PROJECT_ROOT / "manage.py"

    if not manage_py.exists():
        print_status("manage.py not found. Please create a Django project first.")
        return False

    try:
        # Use the venv Python to run Django
        subprocess.run([str(VENV_PYTHON), str(manage_py), "runserver"], check=True)
    except subprocess.CalledProcessError as e:
        print_status(f"Django server error: {e}")
        return False
    except KeyboardInterrupt:
        print_status("Server stopped by user.")

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

    # Step 4: Start Django
    print_status("Setup complete!")
    start_django()


if __name__ == "__main__":
    main()