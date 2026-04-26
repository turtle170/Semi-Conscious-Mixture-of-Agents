use clap::Parser;
use std::fs::{self, File};
use std::io::{Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::windows::named_pipe::ServerOptions;
use flatbuffers::FlatBufferBuilder;

#[allow(dead_code, unused_imports, clippy::all)]
mod messages_generated;

use env::Shard;
use messages_generated::scmoa::{
    Message, MessageArgs, Payload, HivemindUpdate, HivemindUpdateArgs,
    ShardUpdate, ShardUpdateArgs,
};

#[derive(Parser, Debug)]
#[command(name = "aether", version = "2.0", about = "SCMoA Gen 2 Training CLI")]
struct Args {
    #[arg(short, long)]
    svg: PathBuf,

    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    #[arg(short, long, default_value_t = 4096)]
    ram_mb: u64,

    #[arg(short, long)]
    log: PathBuf,
}

struct SVGEnv {
    obstacles: Vec<usvg::Rect>,
    goal: (f32, f32),
}

fn parse_svg(path: &Path) -> SVGEnv {
    let opt = usvg::Options::default();
    let data = fs::read(path).expect("Failed to read SVG file");
    let tree = usvg::Tree::from_data(&data, &opt).expect("Failed to parse SVG");
    
    let mut goal = (90.0, 90.0);
    let mut obstacles = Vec::new();

    // Iterate through all nodes in the tree
    for node in tree.root().children() {
        match node {
            usvg::Node::Path(p) => {
                let b = p.abs_bounding_box();
                if p.id() == "goal" {
                    goal = (b.x() as f32, b.y() as f32);
                } else {
                    obstacles.push(b);
                }
            }
            usvg::Node::Group(g) => {
                for child in g.children() {
                    if let usvg::Node::Path(p) = child {
                        let b = p.abs_bounding_box();
                        if p.id() == "goal" {
                            goal = (b.x() as f32, b.y() as f32);
                        } else {
                            obstacles.push(b);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    
    SVGEnv { obstacles, goal }
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let args = Args::parse();
    
    println!("Aether CLI: Initializing...");
    
    // 1. Resource Calculation
    let mem_per_shard_kb = 256; 
    let available_ram_kb = args.ram_mb * 1024;
    let shard_count = (available_ram_kb / mem_per_shard_kb).min(512) as u32;
    
    let is_performance = (args.threads as f32 / shard_count as f32) > 0.1;
    let hz = if is_performance { 500 } else { 100 };
    
    println!("Aether CLI: Shards={}, Hz={}, Mode={}", 
        shard_count, hz, if is_performance { "P (Performance)" } else { "E (Efficiency)" });

    let _svg_env = parse_svg(&args.svg);
    let mut shards: Vec<Shard> = (0..shard_count).map(|id| {
        let mut s = Shard::new(id);
        if is_performance { s.goal_radius = 5.0; } else { s.goal_radius = 15.0; }
        s
    }).collect();

    let pipe_name = r"\\.\pipe\scmoa_scientist";
    let server = ServerOptions::new().first_pipe_instance(true).create(pipe_name)?;

    println!("Aether CLI: Waiting for Hivemind Agent...");
    server.connect().await?;
    
    let (mut reader, mut writer) = tokio::io::split(server);
    let mut builder = FlatBufferBuilder::with_capacity(16384);
    let mut step_id: u64 = 0;
    let mut last_log_instant = Instant::now();

    loop {
        let loop_start = Instant::now();
        
        let mut shard_data = Vec::with_capacity(shards.len());
        for shard in shards.iter_mut() {
            let (reward, done) = shard.step(1.0 / hz as f32);
            let state = shard.quantize_state();
            let spec_id = if shard.params.gravity > 15.0 { 1 } else { 0 };
            shard_data.push((shard.id, state, reward, done, [shard.params.gravity, shard.params.friction], spec_id));
        }

        builder.reset();
        let mut update_offsets = Vec::new();
        for (id, state, reward, done, ctx, spec_id) in shard_data {
            let context_vec = builder.create_vector(&ctx);
            update_offsets.push(ShardUpdate::create(&mut builder, &ShardUpdateArgs {
                shard_id: id, state, reward, done, context: Some(context_vec), specialist_id: spec_id,
            }));
        }
        let shards_vec = builder.create_vector(&update_offsets);
        let hu = HivemindUpdate::create(&mut builder, &HivemindUpdateArgs { shards: Some(shards_vec), step_id });
        let msg = Message::create(&mut builder, &MessageArgs { payload_type: Payload::HivemindUpdate, payload: Some(hu.as_union_value()) });
        builder.finish(msg, None);
        
        let buf = builder.finished_data();
        writer.write_all(&(buf.len() as u32).to_le_bytes()).await?;
        writer.write_all(buf).await?;

        let mut len_buf = [0u8; 4];
        if reader.read_exact(&mut len_buf).await.is_err() { break; }
        let msg_len = u32::from_le_bytes(len_buf) as usize;
        let mut read_buf = vec![0u8; msg_len];
        reader.read_exact(&mut read_buf).await?;

        let resp_msg = flatbuffers::root::<Message>(&read_buf).unwrap();
        if resp_msg.payload_type() == Payload::HivemindResult {
            let hr = resp_msg.payload_as_hivemind_result().unwrap();
            let results = hr.results().unwrap();
            for i in 0..results.len() {
                let res = results.get(i);
                if (res.shard_id() as usize) < shards.len() {
                    shards[res.shard_id() as usize].apply_action(res.action());
                }
            }
        }

        if last_log_instant.elapsed() >= Duration::from_secs(1) {
            let mut log_file = File::create(&args.log)?;
            writeln!(log_file, "Aether SCMoA Training Log")?;
            writeln!(log_file, "------------------------")?;
            writeln!(log_file, "Step: {}", step_id)?;
            writeln!(log_file, "Shards: {}", shard_count)?;
            writeln!(log_file, "Freq: {} Hz", hz)?;
            writeln!(log_file, "Mode: {}", if is_performance { "P" } else { "E" })?;
            writeln!(log_file, "Uptime: {:?}", last_log_instant.elapsed())?;
            last_log_instant = Instant::now();
        }

        step_id += 1;
        let target_duration = Duration::from_secs_f32(1.0 / hz as f32);
        if let Some(sleep_time) = target_duration.checked_sub(loop_start.elapsed()) {
            tokio::time::sleep(sleep_time).await;
        }
    }

    Ok(())
}

use std::io;
