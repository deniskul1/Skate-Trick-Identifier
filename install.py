import subprocess
import sys
import venv
from pathlib import Path

VENV_DIR = Path("venv")
MIN_PYTHON = (3, 8)
MAX_PYTHON = (3, 11)


def check_python_version():
    version = (sys.version_info.major, sys.version_info.minor)
    if not (MIN_PYTHON <= version <= MAX_PYTHON):
        print(
            f"Warning: you're running Python {version[0]}.{version[1]}, but "
            f"the mediapipe version pinned in requirements.txt only has "
            f"pre-built packages for Python 3.8 through 3.11. Installation "
            f"may fail below — if it does, re-run this with a Python "
            f"3.8-3.11 interpreter (e.g. 'python3.11 install.py').\n"
        )


def create_venv():
    if VENV_DIR.exists():
        print(f"'{VENV_DIR}/' already exists — reusing it.")
        return
    print(f"Creating a virtual environment in '{VENV_DIR}/'...")
    venv.create(VENV_DIR, with_pip=True)


def venv_python_path():
    # Virtual environments lay out their bundled Python interpreter
    # differently on Windows vs. everywhere else.
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_requirements():
    python_path = venv_python_path()
    print("Installing dependencies from requirements.txt — this can take "
          "a few minutes, mediapipe and opencv are large packages...")
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )


def print_next_steps():
    if sys.platform == "win32":
        activate_cmd = r"venv\Scripts\activate"
        run_cmd = "python menu.py"
    else:
        activate_cmd = "source venv/bin/activate"
        run_cmd = "python3 menu.py"

    print("\nSetup complete.\n")
    print("From now on, every time you want to use the project:")
    print(f"  1. Activate the virtual environment:  {activate_cmd}")
    print(f"  2. Run the menu:                      {run_cmd}")
    print("  3. When you're done, type 'deactivate' to leave the "
          "virtual environment.")


def main():
    check_python_version()
    create_venv()
    install_requirements()
    print_next_steps()


if __name__ == "__main__":
    main()
