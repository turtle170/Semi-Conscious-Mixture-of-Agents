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

use env::{Shard, PhysicsParams};
use messages_generated::scmoa::{
    Message, MessageArgs, Payload, HivemindUpdate, HivemindUpdateArgs,
    ShardUpdate, ShardUpdateArgs, HivemindResult, Config, ConfigArgs,
};

#[derive(Parser, Debug, Clone)]
#[command(name = "aether", version = "2.0", about = "Aether SCMoA Hyper-Optimized Training CLI")]
struct Args {
    /// Path to the .SVG environment file
    #[arg(short, long)]
    svg: PathBuf,

    /// Number of CPU threads to utilize
    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    /// Maximum RAM usage in MB
    #[arg(short, long, default_value_t = 4096)]
    ram_mb: u64,

    /// Path to the log file (.txt or .log)
    #[arg(short, long)]
    log: PathBuf,

    // RL Hyperparameters
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,

    #[arg(long, default_value_t = 128)]
    hidden_dim: u32,

    #[arg(long, default_value_t = 8)]
    nhead: u32,

    #[arg(long, default_value_t = 4)]
    num_layers: u32,

    #[arg(long, default_value_t = 4)]
    num_specialists: u32,

    #[arg(long, default_value_t = 16)]
    max_seq: u32,

    #[arg(long, default_value_t = 0.01)]
    entropy_coef: f32,

    #[arg(long, default_value_t = 0.05)]
    mutation_threshold: f32,

    // Environment Parameters
    #[arg(long, default_value = "5.0-25.0")]
    gravity_range: String,

    #[arg(long, default_value = "0.7-1.0")]
    friction_range: String,

    #[arg(long, default_value_t = 1000)]
    checkpoint_freq: u64,
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

fn parse_range(range_str: &str) -> (f32, f32) {
    let parts: Vec<&str> = range_str.split('-').collect();
    if parts.len() == 2 {
        let low = parts[0].parse().unwrap_or(0.0);
        let high = parts[1].parse().unwrap_or(1.0);
        (low, high)
    } else {
        (0.0, 1.0)
    }
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let args = Args::parse();
    println!("Aether CLI: Starting perpetual evolution...");

    // 1. Resource & Mode Calculation
    let mem_per_shard_kb = 512; // Adjusted for Gen 2 Hivemind overhead
    let available_ram_kb = args.ram_mb * 1024;
    let shard_count = (available_ram_kb / mem_per_shard_kb).min(1024) as u32;
    let is_performance = (args.threads as f32 / shard_count as f32) > 0.05;
    let hz = if is_performance { 500 } else { 100 };

    let g_range = parse_range(&args.gravity_range);
    let f_range = parse_range(&args.friction_range);

    let _svg_env = parse_svg(&args.svg);
    let mut shards: Vec<Shard> = (0..shard_count).map(|id| {
        let mut s = Shard::new(id);
        s.goal_radius = if is_performance { 4.0 } else { 10.0 };
        s
    }).collect();

    let pipe_name = r"\\.\pipe\scmoa_scientist";
    let server = ServerOptions::new().first_pipe_instance(true).create(pipe_name)?;
    println!("Aether CLI: Mode={} | Shards={} | Hz={} | MemLimit={}MB", 
             if is_performance { "P" } else { "E" }, shard_count, hz, args.ram_mb);

    println!("Aether CLI: Awaiting Hivemind Agent...");
    server.connect().await?;
    println!("Aether CLI: Agent Linked.");

    let (mut reader, mut writer) = tokio::io::split(server);
    let mut builder = FlatBufferBuilder::with_capacity(32768);

    // 2. Initialization: Send Config
    builder.reset();
    let config = Config::create(&mut builder, &ConfigArgs {
        learning_rate: args.lr,
        batch_size: shard_count, // Shard count matches batch size in Hivemind
        hidden_dim: args.hidden_dim,
        nhead: args.nhead,
        num_layers: args.num_layers,
        num_specialists: args.num_specialists,
        max_seq: args.max_seq,
        entropy_coef: args.entropy_coef,
        mutation_threshold: args.mutation_threshold,
    });
    let msg = Message::create(&mut builder, &MessageArgs {
        payload_type: Payload::Config,
        payload: Some(config.as_union_value()),
    });
    builder.finish(msg, None);
    let config_buf = builder.finished_data();
    writer.write_all(&(config_buf.len() as u32).to_le_bytes()).await?;
    writer.write_all(config_buf).await?;
    writer.flush().await?;

    let mut step_id: u64 = 0;
    let mut last_log_instant = Instant::now();
    let start_time = Instant::now();

    loop {
        let loop_start = Instant::now();
        
        // 3. Step Shards
        let mut shard_data = Vec::with_capacity(shards.len());
        for shard in shards.iter_mut() {
            let (reward, done) = shard.step(1.0 / hz as f32);
            let state = shard.quantize_state();
            
            // Specialist routing based on normalized gravity regime
            let spec_id = if shard.params.gravity > (g_range.0 + (g_range.1 - g_range.0)/2.0) { 1 } else { 0 };
            
            shard_data.push((shard.id, state, reward, done, [shard.params.gravity, shard.params.friction], spec_id));
            
            if done {
                shard.randomize_physics(); // Uses internal rand range
            }
        }

        // 4. Batch Hivemind Update
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

        // 5. Response handling
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

        // 6. Hyper-Optimized 1 FPS Logging (Overwriting)
        if last_log_instant.elapsed() >= Duration::from_secs(1) {
            if let Ok(mut log_file) = File::create(&args.log) {
                let uptime = start_time.elapsed();
                writeln!(log_file, "Aether SCMoA Hyper-Log | Step: {}", step_id).ok();
                writeln!(log_file, "----------------------------------------").ok();
                writeln!(log_file, "Uptime:      {:?}", uptime).ok();
                writeln!(log_file, "Throughput:  {} steps/sec", step_id / (uptime.as_secs().max(1))).ok();
                writeln!(log_file, "Shards:      {}", shard_count).ok();
                writeln!(log_file, "Frequency:   {} Hz", hz).ok();
                writeln!(log_file, "Resource:    {} Threads | {} MB RAM", args.threads, args.ram_mb).ok();
                writeln!(log_file, "Arch:        {} Dim | {} Layers | {} Specialists", args.hidden_dim, args.num_layers, args.num_specialists).ok();
                writeln!(log_file, "Learning:    LR={} | MutThresh={}", args.lr, args.mutation_threshold).ok();
                writeln!(log_file, "Env:         G_Range={:?} | F_Range={:?}", g_range, f_range).ok();
            }
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
