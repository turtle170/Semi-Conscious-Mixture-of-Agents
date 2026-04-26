# Project Aether: SCMoA Gen 2 (Hivemind)

Project Aether is a Reinforcement Learning (RL) research framework implementing the **Semi-Conscious Mixture of Agents (SCMoA)** architecture. In Gen 2, we introduce the **Hivemind** protocol.

## 🚀 Hivemind Architecture

Gen 2 scales the architecture into a massively parallel, multi-specialist system:

*   **Massively Parallel Shards:** The Rust core spawns dozens of parallel physics environments (shards), each with randomized parameters.
*   **The Hivemind Agent (Python/PyTorch):** A unified Transformer backbone that processes batches of states from all shards. Experiences are shared instantly in a unified latent space.
*   **Mixture of Specialists:** Different neural "heads" specialize in specific physics regimes (e.g., high-gravity vs. low-friction).
*   **Intelligent Routing:** The Distiller acts as a high-speed router, analyzing the physics of each shard and directing its 6-bit state to the most qualified specialist head.

## 🛠 Tech Stack (Gen 2 Optimized)

*   **Batch IPC:** Hivemind protocol batches dozens of shard updates into single zero-copy FlatBuffer messages.
*   **SIMD Memory Search:** Rust core uses explicit AVX-512/AVX2 branching for ultra-fast historical context retrieval.
*   **JIT Inference:** Python agents utilize `torch.compile` for hardware-specific optimizations (AVX-VNNI).
*   **High-Frequency Orchestration:** The system maintains a 200Hz+ loop across 32+ parallel shards.

## 📁 Project Structure

```text
aether/
├── schema/             # FlatBuffers communication schemas (.fbs)
├── rust_core/          # High-performance Rust workspace
│   ├── env/            # Custom 2D kinematic physics environment
│   ├── distiller/      # Main orchestrator & IPC server
│   └── fetcher/        # Vector search context retrieval
├── py_agents/          # AI Agent implementations
│   ├── scientist.py    # Transformer World Model
│   ├── worker.py       # PPO Actor-Critic Agent
│   └── telemetry.py    # Real-time dashboard & visualizer
└── README.md
```

## 🔋 Key Features

*   **6-Bit Constraint:** All inter-agent communication is quantized to 64 discrete states, forcing the emergence of efficient latent representations.
*   **Perpetual Evolution:** The system monitors its own accuracy. If it plateaus, the Distiller triggers a `TopologyMutation` (e.g., `ADD_HEAD`), adding attention heads to the neural models on-the-fly.
*   **Curriculum Learning:** The environment scales difficulty (sensor noise and goal precision) as the Worker achieves success streaks.
*   **Cross-Attention Memory:** Agents reconcile low-bandwidth observations with high-dimensional historical context fetched from the Rust memory buffer.

## 🏁 Getting Started

### Prerequisites
*   Rust (Nightly toolchain)
*   Python 3.11+
*   FlatBuffers Compiler (`flatc`)

### Installation
1.  **Clone the Repository:**
    ```powershell
    git clone https://github.com/turtle170/Semi-Conscious-Mixture-of-Agents.git
    cd Semi-Conscious-Mixture-of-Agents
    ```

2.  **Setup Python Environment:**
    ```powershell
    python -m venv py_env
    .\py_env\Scripts\activate
    pip install torch flatbuffers pywin32 matplotlib
    ```

3.  **Compile Schemas:**
    ```powershell
    .\flatc.exe --rust -o rust_core/distiller/src/ schema/messages.fbs
    .\flatc.exe --python -o py_agents/schema/ schema/messages.fbs
    ```

4.  **Run the System:**
    *   Start the **Distiller** (Rust):
        ```powershell
        cd rust_core
        cargo +nightly run -p distiller
        ```
    *   In a separate terminal, start the **Scientist** (Python):
        ```powershell
        .\py_env\Scripts\python.exe py_agents/scientist.py
        ```
    *   (Optional) Start the **Telemetry Dashboard**:
        ```powershell
        .\py_env\Scripts\python.exe py_agents/telemetry.py
        ```

## 📜 License
MIT License. See `LICENSE` for details.
