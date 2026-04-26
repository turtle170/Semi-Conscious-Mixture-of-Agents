import sys
from cx_Freeze import setup, Executable

# Project Metadata
NAME = "Project Aether SCMoA"
VERSION = "2.1"
DESCRIPTION = "Semi-Conscious Mixture of Agents - Gen 2 Orchestrator & Training Suite"
AUTHOR = "Project Aether Team"

# MSI Properties
upgrade_code = "{7E5B3D2A-1C4F-4A9B-8D7E-9F0A1B2C3D4E}"

build_exe_options = {
    "packages": ["os", "sys", "subprocess", "threading", "time", "tkinter"],
    "includes": ["tkinter"],
    "include_files": [
        "flatc.exe",
        "schema/",
        "rust_core/Cargo.toml",
        "rust_core/Cargo.lock",
        "rust_core/distiller/",
        "rust_core/env/",
        "rust_core/fetcher/",
        "py_agents/",
        "README.md",
        ".gitignore"
    ]
}

bdist_msi_options = {
    "upgrade_code": upgrade_code,
    "add_to_path": False,
    "initial_target_dir": rf"[ProgramFilesFolder]\{NAME}",
    "summary_data": {
        "author": AUTHOR,
        "comments": DESCRIPTION
    }
}

base = None
if sys.platform == "win32":
    base = "Win32GUI" # Hides the console window

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options
    },
    executables=[
        Executable(
            "AetherInstaller.py",
            base=base,
            target_name="AetherSetup.exe",
            shortcut_name="Aether SCMoA Installer",
            shortcut_dir="ProgramMenuFolder"
        )
    ]
)
