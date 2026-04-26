#[allow(dead_code, unused_imports, clippy::all)]
mod messages_generated;

use env::Shard;
use fetcher::ContextFetcher;
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use messages_generated::scmoa::{
    Message, MessageArgs, Payload, HivemindUpdate, HivemindUpdateArgs,
    ShardUpdate, ShardUpdateArgs, ShardResult, HivemindResult,
};
use std::collections::HashMap;
use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::windows::named_pipe::ServerOptions;

const NUM_SHARDS: u32 = 32;

#[tokio::main]
async fn main() -> io::Result<()> {
    let pipe_name = r"\\.\pipe\scmoa_scientist";
    let server = ServerOptions::new().first_pipe_instance(true).create(pipe_name)?;

    println!("Distiller: Booting Hivemind (Gen 2) Orchestrator...");
    println!("Distiller: Spawning {} parallel environments...", NUM_SHARDS);
    
    let mut shards: Vec<Shard> = (0..NUM_SHARDS).map(Shard::new).collect();
    let mut fetcher = ContextFetcher::new();
    
    println!("Distiller: Waiting for Hivemind Agent connection...");
    server.connect().await?;
    println!("Distiller: Hivemind Linked.");

    let (mut reader, mut writer) = tokio::io::split(server);
    let mut builder = FlatBufferBuilder::with_capacity(8192);
    let mut step_id: u64 = 0;

    loop {
        // 1. Parallel Shard Steps & Quantization
        let mut shard_data = Vec::new();
        for shard in shards.iter_mut() {
            let (reward, done) = shard.step(0.1);
            let state = shard.quantize_state();
            
            // Router Logic: Pick Specialist based on physics regime
            let spec_id = match (shard.params.gravity >= 15.0, shard.params.friction >= 0.85) {
                (false, false) => 0,
                (false, true)  => 1,
                (true, false)  => 2,
                (true, true)   => 3,
            };

            // Memory Context (Simplified: just use current params for this shard)
            shard_data.push((shard.id, state, reward, done, [shard.params.gravity, shard.params.friction], spec_id));
            
            if done { shard.randomize_physics(); }
        }

        // 2. Build Hivemind Batch Update
        builder.reset();
        let mut update_offsets = Vec::new();
        for (id, state, reward, done, ctx, spec_id) in shard_data {
            let context_vec = builder.create_vector(&ctx);
            let su = ShardUpdate::create(&mut builder, &ShardUpdateArgs {
                shard_id: id,
                state,
                reward,
                done,
                context: Some(context_vec),
                specialist_id: spec_id,
            });
            update_offsets.push(su);
        }
        
        let shards_vec = builder.create_vector(&update_offsets);
        let hu = HivemindUpdate::create(&mut builder, &HivemindUpdateArgs {
            shards: Some(shards_vec),
            step_id,
        });
        
        let msg = Message::create(&mut builder, &MessageArgs {
            payload_type: Payload::HivemindUpdate,
            payload: Some(hu.as_union_value()),
        });
        builder.finish(msg, None);
        let buf = builder.finished_data();
        
        // 3. Send Batch
        writer.write_all(&(buf.len() as u32).to_le_bytes()).await?;
        writer.write_all(buf).await?;
        writer.flush().await?;

        // 4. Receive Batch Inference
        let mut len_buf = [0u8; 4];
        if reader.read_exact(&mut len_buf).await.is_err() { break; }
        let msg_len = u32::from_le_bytes(len_buf) as usize;
        let mut read_buf = vec![0u8; msg_len];
        reader.read_exact(&mut read_buf).await?;

        // 5. Apply Hivemind Results to Shards
        let resp_msg = flatbuffers::root::<Message>(&read_buf).expect("Distiller: Parse Error");
        if resp_msg.payload_type() == Payload::HivemindResult {
            let hr = resp_msg.payload_as_hivemind_result().unwrap();
            let results = hr.results().unwrap();
            
            for i in 0..results.len() {
                let res = results.get(i);
                let sid = res.shard_id() as usize;
                let action = res.action();
                
                if sid < shards.len() {
                    shards[sid].apply_action(action);
                }
            }
        }

        if step_id % 100 == 0 {
            println!("Step {}: Orchestrating {} shards across 4 specialists.", step_id, NUM_SHARDS);
        }

        step_id += 1;
        tokio::time::sleep(std::time::Duration::from_millis(5)).await; // 200Hz throughput
    }

    Ok(())
}
