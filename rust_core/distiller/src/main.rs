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
<<<<<<< HEAD

#[cfg(windows)]
use tokio::net::windows::named_pipe::ServerOptions;

#[cfg(not(windows))]
mod mock_pipe {
    pub struct ServerOptions {}
    impl ServerOptions {
        pub fn new() -> Self { Self {} }
        pub fn first_pipe_instance(&mut self, _: bool) -> &mut Self { self }
        pub fn create(&self, _: &str) -> std::io::Result<tokio::io::DuplexStream> {
            let (client, server) = tokio::io::duplex(1024);
            // Just drop the client and return the server so it compiles
            Ok(server)
        }
    }
}
#[cfg(not(windows))]
use mock_pipe::ServerOptions;

=======
use tokio::net::windows::named_pipe::ServerOptions;
>>>>>>> main

#[tokio::main]
async fn main() -> io::Result<()> {
    let scientist_pipe = r"\\.\pipe\scmoa_scientist";
    let scientist_server = ServerOptions::new().first_pipe_instance(true).create(scientist_pipe)?;

    println!("Distiller: Waiting for Scientist connection...");

    #[cfg(windows)]
    scientist_server.connect().await?;

    println!("Distiller: Scientist connected!");

    let mut env = Environment::new();
    let mut fetcher = ContextFetcher::new();
    let mut builder = FlatBufferBuilder::new();
    let mut accuracy_history: VecDeque<bool> = VecDeque::with_capacity(1000);
    let mut step_history: VecDeque<u8> = VecDeque::with_capacity(16);
    let mut success_streak = 0;
    let mut step_id: u64 = 0;
    
    let (mut reader, mut writer) = tokio::io::split(scientist_server);

    loop {
        let (reward, done) = env.step(0.1);
        let current_quantized_state = env.quantize_state();
        
        if done {
            success_streak += 1;
            println!("Distiller: Goal reached! Streak: {}", success_streak);
        }

        step_history.push_back(current_quantized_state);
        if step_history.len() > 5 { step_history.pop_front(); }

        let current_seq: Vec<u8> = step_history.iter().cloned().collect();
        let context_params = fetcher.query_context(&current_seq);
        fetcher.record(current_seq, env.params);

        let context_vec_data = [context_params.gravity, context_params.friction];

        // Send State
        builder.reset();
        let state_vec = builder.create_vector(&[current_quantized_state]);
        let context_vec = builder.create_vector(&context_vec_data);
        let su = StateUpdate::create(&mut builder, &StateUpdateArgs {
            state: Some(state_vec), step_id, reward, done, context: Some(context_vec),
        });
        let msg = Message::create(&mut builder, &MessageArgs {
            payload_type: Payload::StateUpdate, payload: Some(su.as_union_value()),
        });
        builder.finish(msg, None);
        let buf = builder.finished_data();
        writer.write_all(&(buf.len() as u32).to_le_bytes()).await?;
        writer.write_all(buf).await?;

        // Receive Prediction
        let mut len_buf = [0u8; 4];
        if reader.read_exact(&mut len_buf).await.is_err() { break; }
        let msg_len = u32::from_le_bytes(len_buf) as usize;
        let mut read_buf = vec![0u8; msg_len];
        reader.read_exact(&mut read_buf).await?;

        let resp_msg = flatbuffers::root::<Message>(&read_buf).expect("Parse error");
        if resp_msg.payload_type() == Payload::Prediction {
            let pred = resp_msg.payload_as_prediction().unwrap();
            let predicted_val = pred.predicted_state().unwrap().get(0);
            let is_correct = predicted_val == current_quantized_state;
            
            accuracy_history.push_back(is_correct);
            if accuracy_history.len() > 1000 { accuracy_history.pop_front(); }
            
            if success_streak > 10 {
                println!("Distiller: Streak > 10. CURRICULUM BUMP.");
                builder.reset();
                let path = builder.create_string(&format!("checkpoints/gen1_step_{}.pth", step_id));
                let cp = Checkpoint::create(&mut builder, &CheckpointArgs { filepath: Some(path), step_id });
                let msg = Message::create(&mut builder, &MessageArgs {
                    payload_type: Payload::Checkpoint, payload: Some(cp.as_union_value()),
                });
                builder.finish(msg, None);
                let cp_buf = builder.finished_data();
                writer.write_all(&(cp_buf.len() as u32).to_le_bytes()).await?;
                writer.write_all(cp_buf).await?;
                env.bump_difficulty();
                success_streak = 0;
            }
        }

<<<<<<< HEAD
        if done {
=======
        if done { 
>>>>>>> main
            env.state.position = (50.0, 50.0);
            env.state.velocity = (0.0, 0.0);
        }
        step_id += 1;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    Ok(())
}
