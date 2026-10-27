//! herta-rs — the small Rust helper that ships inside Herta Station.
//!
//! Two jobs, both used by `models.py`:
//!   herta-rs download URL PART [--connections N] [--resume]
//!   herta-rs info FILE
//!
//! `download` splits a Hugging Face blob into N HTTP range requests and writes
//! each slice into its own chunk file, so a 66 GB shard arrives several times
//! faster than a single Python stream and survives Ctrl+C. Progress is printed
//! as JSON lines: `{"done":123,"total":456}` and finally `{"phase":"done"}`.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const VERSION: &str = "0.1.0";
const MIN_SPLIT: u64 = 8 * 1024 * 1024; // below this, splitting only adds overhead
const CHUNK_READ: usize = 256 * 1024;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("download") => {
            if args.len() < 3 {
                usage();
            }
            let url = args[1].clone();
            let part = PathBuf::from(&args[2]);
            let mut connections = 6usize;
            let mut resume = false;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--connections" | "-c" => {
                        i += 1;
                        if i < args.len() {
                            connections = args[i].parse().unwrap_or(6).clamp(1, 16);
                        }
                    }
                    "--resume" => resume = true,
                    other => {
                        eprintln!("herta-rs: unknown flag {other}");
                        usage();
                    }
                }
                i += 1;
            }
            exit(download(&url, &part, connections, resume));
        }
        Some("info") => {
            if args.len() < 2 {
                usage();
            }
            exit(info(&PathBuf::from(&args[1])));
        }
        Some("version") | Some("--version") | Some("-V") => println!("herta-rs {VERSION}"),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!(
        "herta-rs {VERSION}\n\
         \x20 download URL PART [--connections N] [--resume]\n\
         \x20 info FILE"
    );
    exit(2)
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .user_agent(concat!("herta-rs/", "0.1.0"))
        .build()
        .into()
}

/// HEAD the URL to learn the size and whether ranges are supported.
fn probe(agent: &ureq::Agent, url: &str) -> (u64, bool) {
    if let Ok(resp) = agent.head(url).call() {
        let len = header_u64(&resp, "content-length");
        let ranges = header_eq(&resp, "accept-ranges", "bytes");
        if len > 0 {
            return (len, ranges);
        }
    }
    (0, false)
}

fn header_u64(resp: &ureq::http::Response<ureq::Body>, name: &str) -> u64 {
    resp.headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn header_eq(resp: &ureq::http::Response<ureq::Body>, name: &str, want: &str) -> bool {
    resp.headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim().eq_ignore_ascii_case(want))
        .unwrap_or(false)
}

fn download(url: &str, part: &Path, connections: usize, resume: bool) -> i32 {
    let agent = agent();
    if let Some(dir) = part.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let (total, ranges) = probe(&agent, url);
    if total == 0 || !ranges || total < MIN_SPLIT || connections == 1 {
        return single(&agent, url, part, total, resume);
    }

    let per = total / connections as u64;
    let mut plan: Vec<(u64, u64, PathBuf)> = Vec::with_capacity(connections);
    for i in 0..connections {
        let start = i as u64 * per;
        let end = if i + 1 == connections { total } else { (i as u64 + 1) * per };
        plan.push((start, end, chunk_path(part, i)));
    }

    let done = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    for (start, end, path) in &plan {
        let have = if resume {
            fs::metadata(path).map(|m| m.len()).unwrap_or(0)
        } else {
            let _ = fs::remove_file(path);
            0
        };
        let have = have.min(end - start);
        done.fetch_add(have, Ordering::Relaxed);
    }

    let reported = done.load(Ordering::Relaxed);
    let mut handles = Vec::new();
    for (start, end, path) in plan.clone() {
        let agent = agent.clone();
        let url = url.to_string();
        let done = Arc::clone(&done);
        let stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || -> Result<(), String> {
            let have = if resume {
                fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
            } else {
                0
            };
            let have = have.min(end - start);
            let from = start + have;
            if from >= end {
                return Ok(());
            }
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let resp = agent
                .get(&url)
                .header("Range", &format!("bytes={from}-{}", end - 1))
                .call()
                .map_err(|e| format!("range request failed: {e}"))?;
            let mut body = resp.into_body();
            let mut reader = body.as_reader();
            let mut buf = vec![0u8; CHUNK_READ];
            loop {
                if stop.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                let read = match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(format!("read failed: {e}")),
                };
                file.write_all(&buf[..read])
                    .map_err(|e| format!("write failed: {e}"))?;
                done.fetch_add(read as u64, Ordering::Relaxed);
            }
            Ok(())
        }));
    }

    let start_at = Instant::now();
    while handles.iter().any(|h| !h.is_finished()) {
        thread::sleep(Duration::from_millis(250));
        let now = done.load(Ordering::Relaxed);
        let elapsed = start_at.elapsed().as_secs_f64().max(0.001);
        let speed = (now.saturating_sub(reported)) as f64 / elapsed;
        println!(
            "{{\"done\":{now},\"total\":{total},\"speed\":{:.0}}}",
            speed
        );
        let _ = std::io::stdout().flush();
    }

    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                eprintln!("herta-rs: {err}");
                return 1;
            }
            Err(_) => {
                eprintln!("herta-rs: worker panicked");
                return 1;
            }
        }
    }

    // Stitch the slices back together into the .part file.
    let mut out = match OpenOptions::new().create(true).write(true).truncate(true).open(part) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("herta-rs: cannot write {}: {e}", part.display());
            return 1;
        }
    };
    let mut written = 0u64;
    for (_, _, path) in &plan {
        match fs::File::open(path) {
            Ok(mut chunk) => {
                if let Err(e) = std::io::copy(&mut chunk, &mut out) {
                    eprintln!("herta-rs: merge failed: {e}");
                    return 1;
                }
                written += fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            }
            Err(e) => {
                eprintln!("herta-rs: missing slice {}: {e}", path.display());
                return 1;
            }
        }
    }
    if let Err(e) = out.flush() {
        eprintln!("herta-rs: flush failed: {e}");
        return 1;
    }
    if written != total {
        eprintln!("herta-rs: size mismatch {written}/{total}");
        return 1;
    }
    for (_, _, path) in &plan {
        let _ = fs::remove_file(path);
    }
    println!("{{\"phase\":\"done\",\"done\":{written},\"total\":{total}}}");
    0
}

fn single(agent: &ureq::Agent, url: &str, part: &Path, total: u64, resume: bool) -> i32 {
    let have = if resume {
        fs::metadata(part).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };
    if total > 0 && have == total {
        println!("{{\"phase\":\"done\",\"done\":{have},\"total\":{total}}}");
        return 0;
    }
    let mut request = agent.get(url);
    if have > 0 {
        request = request.header("Range", &format!("bytes={have}-"));
    }
    let resp = match request.call() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("herta-rs: request failed: {e}");
            return 1;
        }
    };
    let mut body = resp.into_body();
    let mut reader = body.as_reader();
    let mut file = match OpenOptions::new().create(true).append(true).open(part) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("herta-rs: cannot write {}: {e}", part.display());
            return 1;
        }
    };
    let mut done = have;
    let start_at = Instant::now();
    let mut last = Instant::now();
    let mut buf = vec![0u8; CHUNK_READ];
    loop {
        let read = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                eprintln!("herta-rs: read failed: {e}");
                return 1;
            }
        };
        if let Err(e) = file.write_all(&buf[..read]) {
            eprintln!("herta-rs: write failed: {e}");
            return 1;
        }
        done += read as u64;
        if last.elapsed() >= Duration::from_millis(250) {
            let speed = done as f64 / start_at.elapsed().as_secs_f64().max(0.001);
            println!("{{\"done\":{done},\"total\":{total},\"speed\":{speed:.0}}}");
            let _ = std::io::stdout().flush();
            last = Instant::now();
        }
    }
    if let Err(e) = file.flush() {
        eprintln!("herta-rs: flush failed: {e}");
        return 1;
    }
    if total > 0 && done != total {
        eprintln!("herta-rs: size mismatch {done}/{total}");
        return 1;
    }
    println!("{{\"phase\":\"done\",\"done\":{done},\"total\":{total}}}");
    0
}

fn chunk_path(part: &Path, index: usize) -> PathBuf {
    let mut name = part.as_os_str().to_os_string();
    name.push(format!(".c{index}"));
    PathBuf::from(name)
}

/// Minimal GGUF header reader: magic + version + tensor count.
fn info(path: &Path) -> i32 {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("herta-rs: cannot open {}: {e}", path.display());
            return 1;
        }
    };
    let mut head = [0u8; 24];
    if file.read_exact(&mut head).is_err() {
        eprintln!("herta-rs: file too small to be a GGUF");
        return 1;
    }
    let magic = &head[0..4];
    let version = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
    let tensors = u64::from_le_bytes([
        head[8], head[9], head[10], head[11], head[12], head[13], head[14], head[15],
    ]);
    let kv = u64::from_le_bytes([
        head[16], head[17], head[18], head[19], head[20], head[21], head[22], head[23],
    ]);
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let ok = magic == b"GGUF";
    println!(
        "{{\"path\":{},\"gguf\":{ok},\"version\":{version},\"tensors\":{tensors},\"kv\":{kv},\"bytes\":{size}}}",
        json_string(&path.display().to_string())
    );
    if ok {
        0
    } else {
        1
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
