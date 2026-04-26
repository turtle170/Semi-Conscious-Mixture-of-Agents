# Project Aether: SCMoA Gen 2 (Hivemind)

Project Aether is a Reinforcement Learning (RL) research framework implementing the **Semi-Conscious Mixture of Agents (SCMoA)** architecture. It scales RL training into a massively parallel, hardware-aware ecosystem.

## 🚀 Hivemind Architecture

*   **Massively Parallel Shards:** Rust-driven physics environments running at up to 500Hz.
*   **Unified Latent Space:** Multiple agents share knowledge through a centralized Hivemind backbone.
*   **Mixture of Specialists:** Specialist neural heads evolve to handle specific physics regimes.
*   **Hardware Native:** Explicit SIMD (AVX-512) and JIT (AVX-VNNI) optimizations for modern CPUs.

## 📖 User Manual: Aether CLI (`aether`)

Aether Gen 2 includes a hyper-optimized CLI for direct training control and multi-format model exporting.

### 🏁 Quick Start

1.  **Build the CLI:**
    ```powershell
    cd rust_core
    cargo +nightly build --release -p distiller --bin aether
    ```
2.  **Start Training:**
    ```powershell
    .\target\release\aether.exe --svg env.svg --log training.log --ram-mb 8192
    ```

### ⚙️ CLI Command Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--svg <PATH>` | (Required) | Path to the .SVG file defining the environment geometry. |
| `--log <PATH>` | (Required) | Path to the status log (overwritten at 1 FPS). |
| `--output-dir <DIR>` | `outputs/` | Directory where trained models and checkpoints are saved. |
| `--output-format <FMT>`| `SafeTensors`| Model format: `PyTorch`, `SafeTensors`, `ONNX`, `GGUF`, `EXL2`, `AWQ`, `TF`, `TFLite`. |
| `--quantization <TYPE>`| `None` | Quantization method: `INT8`, `FP16`, `AWQ`, `Q4_K_M`, etc. |
| `--checkpoint-time <SEC>`| `300` | Interval in seconds between automatic model saves. |
| `--threads <INT>` | `8` | Number of CPU threads to allocate to physics shards. |
| `--ram-mb <INT>` | `4096` | RAM limit. Aether auto-calculates shard density based on this. |
| `--lr <FLOAT>` | `1e-4` | Optimizer learning rate. |
| `--hidden-dim <INT>` | `128` | Latent dimension size of the Transformer backbone. |
| `--nhead <INT>` | `8` | Number of attention heads in the world model. |
| `--num-layers <INT>` | `4` | Number of Transformer layers. |
| `--num-specialists <INT>` | `4` | Number of specialist mixture heads in the Hivemind. |
| `--max-seq <INT>` | `16` | Maximum sequence length for temporal context. |
| `--gravity-range <STR>` | `5.0-25.0` | Gravity randomization bounds (Low-High). |
| `--friction-range <STR>` | `0.7-1.0` | Friction randomization bounds (Low-High). |
| `--mutation-threshold <FLOAT>`| `0.05` | Accuracy floor that triggers topology expansion. |

### 🛠 Environment Design (.SVG)
Aether parses standard SVG files:
*   **Obstacles:** Standard `<path>` or `<rect>` elements are treated as rigid obstacles.
*   **Goal:** Any element with `id="goal"` is treated as the target reward zone.

## 📦 Installer

A dedicated GUI installer `AetherInstaller.py` is provided to automate the setup of Rust, Python, and all system dependencies.

```powershell
# Run the installer
python AetherInstaller.py
```

## 📜 License
MIT License.
