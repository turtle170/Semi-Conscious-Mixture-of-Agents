import os
import sys
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Aether SCMoA Installer (Gen 2)

class AetherInstaller:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Aether: SCMoA Installer")
        self.root.geometry("600x500")
        self.root.configure(bg="#1e1e1e")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TLabel", foreground="white", background="#1e1e1e", font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        
        self.main_frame = tk.Frame(root, bg="#1e1e1e", padx=30, pady=30)
        self.main_frame.pack(fill="both", expand=True)
        
        # Header
        self.header = ttk.Label(self.main_frame, text="PROJECT AETHER", style="Header.TLabel")
        self.header.pack(pady=(0, 10))
        
        self.subhead = ttk.Label(self.main_frame, text="Semi-Conscious Mixture of Agents - Gen 2")
        self.subhead.pack(pady=(0, 30))
        
        # Options
        self.auto_install_var = tk.BooleanVar(value=True)
        self.auto_install_cb = tk.Checkbutton(
            self.main_frame, text="Automatically Install Required Tools (Rust, Python, Deps)", 
            variable=self.auto_install_var, bg="#1e1e1e", fg="white", selectcolor="#333",
            activebackground="#1e1e1e", activeforeground="white", font=("Segoe UI", 10)
        )
        self.auto_install_cb.pack(anchor="w", pady=5)
        
        self.advanced_btn = ttk.Button(self.main_frame, text="Advanced Installation Settings", command=self.toggle_advanced)
        self.advanced_btn.pack(pady=10)
        
        self.advanced_frame = tk.Frame(self.main_frame, bg="#2d2d2d", padx=10, pady=10)
        self.advanced_visible = False
        
        self.install_path_label = ttk.Label(self.advanced_frame, text="Installation Path:", background="#2d2d2d")
        self.install_path_label.pack(anchor="w")
        
        self.path_entry = ttk.Entry(self.advanced_frame, width=50)
        self.path_entry.insert(0, os.getcwd())
        self.path_entry.pack(side="left", padx=(0, 5))
        
        self.browse_btn = ttk.Button(self.advanced_frame, text="Browse", command=self.browse_path)
        self.browse_btn.pack(side="left")
        
        # Status
        self.status_label = ttk.Label(self.main_frame, text="Ready to initialize.")
        self.status_label.pack(pady=(20, 5))
        
        self.progress = ttk.Progressbar(self.main_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=5)
        
        self.log_text = tk.Text(self.main_frame, height=8, bg="#000", fg="#0f0", font=("Consolas", 8))
        self.log_text.pack(fill="both", expand=True, pady=10)
        
        # Install Button
        self.install_btn = ttk.Button(self.main_frame, text="START INSTALLATION", command=self.start_install)
        self.install_btn.pack(pady=20)

    def toggle_advanced(self):
        if self.advanced_visible:
            self.advanced_frame.pack_forget()
        else:
            self.advanced_frame.pack(fill="x", pady=10)
        self.advanced_visible = not self.advanced_visible

    def browse_path(self):
        path = filedialog.askdirectory()
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def update_status(self, msg, progress):
        self.status_label.config(text=msg)
        self.progress["value"] = progress
        self.root.update_idletasks()

    def start_install(self):
        self.install_btn.config(state="disabled")
        threading.Thread(target=self.run_installation, daemon=True).start()

    def run_installation(self):
        try:
            self.log(">>> Initiating Aether Gen 2 Deployment...")
            
            if self.auto_install_var.get():
                # Check Rust
                self.update_status("Checking for Rust toolchain...", 10)
                res = subprocess.run(["rustc", "--version"], capture_output=True)
                if res.returncode != 0:
                    self.log("Rust not found. Downloading rustup-init.exe...")
                    subprocess.run(["curl.exe", "-L", "-o", "rustup-init.exe", "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"])
                    self.log("Installing Rust (Nightly required)...")
                    subprocess.run([".\\rustup-init.exe", "-y", "--default-toolchain", "nightly"])
                else:
                    self.log("Rust detected: " + res.stdout.decode().strip())

                # Check Python
                self.update_status("Verifying Python environment...", 30)
                res = subprocess.run(["py", "--version"], capture_output=True)
                if res.returncode != 0:
                    self.log("Python launcher not found. Please install Python 3.11+ manually.")
                else:
                    self.log("Python detected.")

            # Setup Project
            target_dir = self.path_entry.get()
            os.chdir(target_dir)
            
            self.update_status("Compiling Rust core (aether cli)...", 50)
            self.log("Building Aether Distiller in Release mode...")
            subprocess.run(["cargo", "+nightly", "build", "--release", "-p", "distiller", "--bin", "aether"], shell=True)
            
            self.update_status("Setting up Python Hivemind...", 70)
            self.log("Creating virtual environment...")
            subprocess.run(["py", "-m", "venv", "py_env"], shell=True)
            
            self.log("Installing AI dependencies (PyTorch, FlatBuffers, etc.)...")
            subprocess.run([".\\py_env\\Scripts\\python.exe", "-m", "pip", "install", "torch", "flatbuffers", "pywin32", "safetensors", "onnx", "numpy"], shell=True)
            
            self.update_status("Finalizing...", 90)
            self.log("Deploying FlatBuffers schema...")
            subprocess.run([".\\flatc.exe", "--rust", "-o", "rust_core/distiller/src/", "schema/messages.fbs"], shell=True)
            subprocess.run([".\\flatc.exe", "--python", "-o", "py_agents/schema/", "schema/messages.fbs"], shell=True)
            
            self.update_status("Installation Complete!", 100)
            self.log("\n>>> PROJECT AETHER READY.")
            self.log("Use: .\\target\\release\\aether.exe --help")
            
            messagebox.showinfo("Success", "Project Aether has been successfully installed and optimized for your hardware.")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Installation Failed", str(e))
        finally:
            self.install_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = AetherInstaller(root)
    root.mainloop()
