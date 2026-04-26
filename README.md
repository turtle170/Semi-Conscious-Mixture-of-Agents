# Project Aether: SCMoA (Semi-Conscious Mixture of Agents)

Project Aether is a Reinforcement Learning (RL) research framework implementing the **Semi-Conscious Mixture of Agents (SCMoA)** architecture. It explores the behavior and evolution of AI agents constrained to communicate via extreme low-bandwidth channels (6-bit quantized signals) while operating in a high-fidelity physics environment.

## 🚀 Architecture Overview

SCMoA is built on a hybrid, high-performance stack designed for low-latency feedback loops and evolutionary neural topology:

*   **The Scientist (Python/PyTorch):** A Transformer-based world model with Cross-Attention. It ingests 6-bit state sequences and historical physics context to predict future environment states.
*   **The Worker (Python/PyTorch):** An Actor-Critic RL policy (PPO-based) that acts as the "probe" in the environment, making decisions based on the Scientist's internal latent representations.
*   **The Distiller (Rust):** The high-speed orchestrator. It manages the physics simulation, performs 6-bit quantization, handles IPC via Windows Named Pipes, and triggers "Topology Mutations" (evolutionary growth) based on performance telemetry.
*   **Mass Context Fetcher (Rust):** A vector search memory system that retrieves high-dimensional physics parameters (gravity, friction) based on 6-bit sequence similarity to resolve environment ambiguity.

## 🛠 Tech Stack

*   **Core Logic:** [Rust](https://www.rust-lang.org/) (Nightly)
*   **AI/ML:** [PyTorch](https://pytorch.org/) (CPU-optimized)
*   **Serialization:** [FlatBuffers](https://google.github.io/flatbuffers/) (Zero-copy IPC)
*   **IPC:** Windows Named Pipes
*   **Concurrency:** [Tokio](https://tokio.rs/) (Async Rust)

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
