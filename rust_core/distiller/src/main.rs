#[allow(dead_code, unused_imports, clippy::all)]
mod messages_generated;

use env::Environment;
use fetcher::ContextFetcher;
use flatbuffers::FlatBufferBuilder;
use messages_generated::scmoa::{
    Message, MessageArgs, Payload, StateUpdate, StateUpdateArgs,
    Checkpoint, CheckpointArgs,
};
use std::collections::VecDeque;
use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::windows::named_pipe::ServerOptions;

#[tokio::main]
async fn main() -> io::Result<()> {
    let scientist_pipe = r"\\.\pipe\scmoa_scientist";
    let scientist_server = ServerOptions::new().first_pipe_instance(true).create(scientist_pipe)?;

    println!("Distiller: Waiting for SCMoA Agent connection...");
    scientist_server.connect().await?;
    println!("Distiller: Agent synchronized.");

    let mut env = Environment::new();
    let mut fetcher = ContextFetcher::new();
    let mut builder = FlatBufferBuilder::with_capacity(1024);
    let mut accuracy_history: VecDeque<bool> = VecDeque::with_capacity(1000);
    let mut step_history: VecDeque<u8> = VecDeque::with_capacity(16);
    let mut success_streak = 0;
    let mut step_id: u64 = 0;
    
    let (mut reader, mut writer) = tokio::io::split(scientist_server);

    loop {
        // 1. Environmental Reality
        let (reward, done) = env.step(0.1);
        let current_state = env.quantize_state();
        
        step_history.push_back(current_state);
        if step_history.len() > 8 { step_history.pop_front(); }

        // 2. Context Retrieval (Memory Search)
        let seq: Vec<u8> = step_history.iter().cloned().collect();
        let ctx_params = fetcher.query_context(&seq);
        fetcher.record(seq, env.params);

        // 3. Dispatch Reality to Agent
        builder.reset();
        let state_vec = builder.create_vector(&[current_state]);
        let ctx_vec = builder.create_vector(&[ctx_params.gravity, ctx_params.friction]);
        let su = StateUpdate::create(&mut builder, &StateUpdateArgs {
            state: Some(state_vec), step_id, reward, done, context: Some(ctx_vec),
        });
        let msg = Message::create(&mut builder, &MessageArgs {
            payload_type: Payload::StateUpdate, payload: Some(su.as_union_value()),
        });
        builder.finish(msg, None);
        let buf = builder.finished_data();
        writer.write_all(&(buf.len() as u32).to_le_bytes()).await?;
        writer.write_all(buf).await?;

        // 4. Ingest Agent Hypothesis (InferenceResult)
        let mut len_buf = [0u8; 4];
        if reader.read_exact(&mut len_buf).await.is_err() { break; }
        let msg_len = u32::from_le_bytes(len_buf) as usize;
        let mut read_buf = vec![0u8; msg_len];
        reader.read_exact(&mut read_buf).await?;

        let resp_msg = flatbuffers::root::<Message>(&read_buf).expect("Orchestrator: Parse error");
        if resp_msg.payload_type() == Payload::InferenceResult {
            let res = resp_msg.payload_as_inference_result().unwrap();
            
            // a. Process Prediction (The Scientist)
            let predicted_val = res.predicted_state().unwrap().get(0);
            let is_correct = predicted_val == current_state;
            accuracy_history.push_back(is_correct);
            if accuracy_history.len() > 1000 { accuracy_history.pop_front(); }
            
            // b. Process Action (The Worker)
            let action_val = res.action().unwrap().get(0);
            env.apply_action(action_val);
            
            // c. Evolution Management
            if done {
                success_streak += 1;
                println!("Step {}: Success! Streak: {}", step_id, success_streak);
                
                if success_streak >= 10 {
                    println!("Distiller: Agent Mastered Physics. Advancing Curriculum...");
                    // Evolutionary Checkpoint
                    builder.reset();
                    let path = builder.create_string(&format!("checkpoints/scmoa_gen1_{}.pth", step_id));
                    let cp = Checkpoint::create(&mut builder, &CheckpointArgs { filepath: Some(path), step_id });
                    let cp_msg = Message::create(&mut builder, &MessageArgs {
                        payload_type: Payload::Checkpoint, payload: Some(cp.as_union_value()),
                    });
                    builder.finish(cp_msg, None);
                    let cp_buf = builder.finished_data();
                    writer.write_all(&(cp_buf.len() as u32).to_le_bytes()).await?;
                    writer.write_all(cp_buf).await?;
                    
                    env.bump_difficulty();
                    env.randomize_physics();
                    success_streak = 0;
                }
                
                // Reset trial position
                env.state.position = (50.0, 50.0);
                env.state.velocity = (0.0, 0.0);
            }
        }

        step_id += 1;
        // Hyper-frequency loop: 10ms steps (100Hz)
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    Ok(())
}
