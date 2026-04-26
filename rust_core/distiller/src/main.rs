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

use env::{Shard, PhysicsParams, Rect};
use messages_generated::scmoa::{
    Message, MessageArgs, Payload, HivemindUpdate, HivemindUpdateArgs,
    ShardUpdate, ShardUpdateArgs, HivemindResult, Config, ConfigArgs,
    Checkpoint, CheckpointArgs,
};

#[derive(Parser, Debug, Clone)]
#[command(name = "aether", version = "2.2", about = "Aether SCMoA Hyper-Optimized Training CLI")]
struct Args {
    /// Path to the .SVG environment file (Optional - generates random if omitted)
    #[arg(short, long)]
    svg: Option<PathBuf>,

    /// Number of CPU threads to utilize
    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    /// Maximum RAM usage in MB
    #[arg(short, long, default_value_t = 4096)]
    ram_mb: u64,

    /// Path to the log file (.txt or .log)
    #[arg(short, long)]
    log: PathBuf,

    /// Directory for model outputs
    #[arg(short, long, default_value = "outputs/")]
    output_dir: PathBuf,

    /// Output format (PyTorch, SafeTensors, ONNX, GGUF, EXL2, AWQ, TF, TFLite)
    #[arg(long, default_value = "SafeTensors")]
    output_format: String,

    /// Quantization type (None, INT8, FP16, AWQ, etc.)
    #[arg(short, long, default_value = "None")]
    quantization: String,

    /// Time in seconds between checkpoints
    #[arg(long, default_value_t = 300)]
    checkpoint_time: u32,

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
}

struct SVGEnv {
    obstacles: Vec<Rect>,
    goal: (f32, f32),
}

fn parse_svg(path: &Path) -> SVGEnv {
    let opt = usvg::Options::default();
    let data = fs::read(path).expect("Failed to read SVG file");
    let tree = usvg::Tree::from_data(&data, &opt).expect("Failed to parse SVG");
    
    let mut goal = (90.0, 90.0);
    let mut obstacles = Vec::new();

    let size = tree.size();
    let scale_x = 100.0 / size.width() as f32;
    let scale_y = 100.0 / size.height() as f32;

    fn traverse(node: &usvg::Group, scale_x: f32, scale_y: f32, obstacles: &mut Vec<Rect>, goal: &mut (f32, f32)) {
        for child in node.children() {
            match child {
                usvg::Node::Path(p) => {
                    let b = p.abs_bounding_box();
                    let nx = b.x() as f32 * scale_x;
                    let ny = b.y() as f32 * scale_y;
                    let nw = b.width() as f32 * scale_x;
                    let nh = b.height() as f32 * scale_y;

                    if p.id() == "goal" {
                        *goal = (nx + nw/2.0, ny + nh/2.0);
                    } else {
                        obstacles.push(Rect { x: nx, y: ny, w: nw, h: nh });
                    }
                }
                usvg::Node::Group(g) => {
                    traverse(g, scale_x, scale_y, obstacles, goal);
                }
                _ => {}
            }
        }
    }

    traverse(tree.root(), scale_x, scale_y, &mut obstacles, &mut goal);
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
    println!("Aether CLI v2.2: Initializing...");

    let mem_per_shard_kb = 512; 
    let available_ram_kb = args.ram_mb * 1024;
    let shard_count = (available_ram_kb / mem_per_shard_kb).min(1024) as u32;
    let is_performance = (args.threads as f32 / shard_count as f32) > 0.05;
    let hz = if is_performance { 500 } else { 100 };

    let g_range = parse_range(&args.gravity_range);

    // Environment Setup
    let svg_env = args.svg.as_ref().map(|path| parse_svg(path));
    
    let mut shards: Vec<Shard> = (0..shard_count).map(|id| {
        let mut s = Shard::new(id);
        if let Some(ref env) = svg_env {
            s.goal_pos = env.goal;
            s.obstacles = env.obstacles.clone();
        } else {
            s.generate_procedural_map();
        }
        s.goal_radius = if is_performance { 4.0 } else { 8.0 };
        s
    }).collect();

    let pipe_name = r"\\.\pipe\scmoa_scientist";
    let server = ServerOptions::new().first_pipe_instance(true).create(pipe_name)?;
    println!("Aether CLI: Shards={} | Hz={} | Mode={} | Env={}", 
             shard_count, hz, if is_performance { "P" } else { "E" },
             if args.svg.is_some() { "Custom SVG" } else { "Procedural" });

    println!("Aether CLI: Awaiting Hivemind Agent...");
    server.connect().await?;
    println!("Aether CLI: Agent Linked.");

    let (mut reader, mut writer) = tokio::io::split(server);
    let mut builder = FlatBufferBuilder::with_capacity(32768);

    builder.reset();
    let format_off = builder.create_string(&args.output_format);
    let quant_off = builder.create_string(&args.quantization);
    let config = Config::create(&mut builder, &ConfigArgs {
        learning_rate: args.lr,
        batch_size: shard_count,
        hidden_dim: args.hidden_dim,
        nhead: args.nhead,
        num_layers: args.num_layers,
        num_specialists: args.num_specialists,
        max_seq: args.max_seq,
        entropy_coef: args.entropy_coef,
        mutation_threshold: args.mutation_threshold,
        output_format: Some(format_off),
        quantization: Some(quant_off),
        checkpoint_time: args.checkpoint_time,
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
    let mut last_checkpoint_instant = Instant::now();
    let start_time = Instant::now();

    loop {
        let loop_start = Instant::now();
        
        let mut shard_data = Vec::with_capacity(shards.len());
        for shard in shards.iter_mut() {
            let (reward, done) = shard.step(1.0 / hz as f32);
            let state = shard.quantize_state();
            let spec_id = if shard.params.gravity > (g_range.0 + (g_range.1 - g_range.0)/2.0) { 1 } else { 0 };
            shard_data.push((shard.id, state, reward, done, [shard.params.gravity, shard.params.friction], spec_id));
            
            if done { 
                shard.randomize_physics(); 
                // If it was procedural, regenerate sometimes to force high generalization
                if args.svg.is_none() && step_id % 1000 == 0 {
                    shard.generate_procedural_map();
                }
            }
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

        if last_checkpoint_instant.elapsed() >= Duration::from_secs(args.checkpoint_time as u64) {
            builder.reset();
            let filename = format!("checkpoint_{}.bin", step_id);
            let path_str = args.output_dir.join(filename).to_string_lossy().to_string();
            let path_off = builder.create_string(&path_str);
            let cp = Checkpoint::create(&mut builder, &CheckpointArgs { filepath: Some(path_off), step_id });
            let msg = Message::create(&mut builder, &MessageArgs { payload_type: Payload::Checkpoint, payload: Some(cp.as_union_value()) });
            builder.finish(msg, None);
            let cp_buf = builder.finished_data();
            writer.write_all(&(cp_buf.len() as u32).to_le_bytes()).await?;
            writer.write_all(cp_buf).await?;
            last_checkpoint_instant = Instant::now();
        }

        if last_log_instant.elapsed() >= Duration::from_secs(1) {
            if let Ok(mut log_file) = File::create(&args.log) {
                let uptime = start_time.elapsed();
                writeln!(log_file, "Aether SCMoA CLI v2.2 | Step: {}", step_id).ok();
                writeln!(log_file, "----------------------------------------").ok();
                writeln!(log_file, "Uptime:      {:?}", uptime).ok();
                writeln!(log_file, "Env:         {}", if args.svg.is_some() { "Custom (Jitter Active)" } else { "Procedural (Auto-Gen)" }).ok();
                writeln!(log_file, "Throughput:  {} steps/sec", step_id / (uptime.as_secs().max(1))).ok();
                writeln!(log_file, "Frequency:   {} Hz | Shards: {}", hz, shard_count).ok();
                writeln!(log_file, "Format:      {} | Quant: {}", args.output_format, args.quantization).ok();
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
