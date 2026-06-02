use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use hexroll3_scroll::{
    frame::FrameConvertor,
    instance::{SandboxInstance},
    renderer::render_entity,
};
use std::path::PathBuf;
use std::collections::HashSet;

#[derive(Parser)]
#[command(name = "hexroll3-cli")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(long)] scroll_dir: PathBuf,
        #[arg(long)] sandbox:    PathBuf,
        #[arg(long)] output:     PathBuf,
        #[arg(long)] seed:       Option<u64>,
    },
    Export {
        #[arg(long)] scroll_dir: PathBuf,
        #[arg(long)] sandbox:    PathBuf,
        #[arg(long)] output:     PathBuf,
    },
}


/// Recursively collect UUIDs from JSON values (for following references)
fn collect_uids_recursive(val: &serde_json::Value, uids: &mut Vec<String>, depth: u32) {
    if depth == 0 { return; }
    match val {
        serde_json::Value::String(s) if s.len() == 8 => {
            // 8-char alphanumeric = likely a UID
            if s.chars().all(|c| c.is_alphanumeric()) {
                uids.push(s.clone());
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                collect_uids_recursive(item, uids, depth - 1);
            }
        }
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if k == "UUID" || k == "uuid" || k == "uid" {
                    collect_uids_recursive(v, uids, depth);
                } else {
                    collect_uids_recursive(v, uids, depth - 1);
                }
            }
        }
        _ => {}
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate { scroll_dir, sandbox, output, seed } => {
            let mut instance = SandboxInstance::new();
            if let Some(s) = seed { instance.with_seed(s); }
            instance.with_scroll(scroll_dir.join("main.scroll"))?;
            eprintln!("[hexroll3-cli] Generating sandbox… (seed: {:?})", seed);
            instance.create(sandbox.to_str().unwrap())?;
            export_all(instance, &output)?;
        }
        Commands::Export { scroll_dir, sandbox, output } => {
            let mut instance = SandboxInstance::new();
            instance.with_scroll(scroll_dir.join("main.scroll"))?;
            instance.open(sandbox.to_str().unwrap())?;
            export_all(instance, &output)?;
        }
    }
    Ok(())
}

fn export_all(instance: SandboxInstance, output: &PathBuf) -> Result<()> {
    let sid = instance.sid().ok_or_else(|| anyhow!("No sandbox ID"))?;
    eprintln!("[hexroll3-cli] Sandbox ID: {}", sid);

    // Clone the instance so we can use both instance and its blueprint independently
    // (SandboxInstance::clone shares the same Arc<Mutex<SandboxBlueprint>>)
    let render_instance = instance.clone();
    
    let exported = instance.repo.inspect(|tx| {
        // Lock the blueprint for rendering
        let mut bp = render_instance.blueprint.lock()
            .map_err(|_| anyhow!("blueprint lock"))?;

        let root_val = tx.load("root")?;
        let realm_uid = root_val.value.as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("root not string"))?;
        eprintln!("[hexroll3-cli] Realm UID: {}", realm_uid);

        let frame_uid = format!("{}_frame", realm_uid);
        let mut frame_val = tx.load(&frame_uid)?;
        let frame = frame_val.value.as_frame();

        let mut all_uids: Vec<String> = Vec::new();
        if let Some(coll_map) = frame.obj["$collections"]["$unused"].as_object() {
            for (name, uids_val) in coll_map {
                if let Some(arr) = uids_val.as_array() {
                    eprintln!("[hexroll3-cli] Collection '{}': {}", name, arr.len());
                    for v in arr {
                        if let Some(s) = v.as_str() { all_uids.push(s.to_string()); }
                    }
                }
            }
        }
        // Also follow UUID references in indexed entities for deeper content
        for uid in all_uids.clone() {
            if let Ok(raw) = tx.load(&uid) {
                collect_uids_recursive(&raw.value, &mut all_uids, 3);
            }
        }
        all_uids.push(realm_uid);
        all_uids.dedup();
        eprintln!("[hexroll3-cli] Rendering {} entities…", all_uids.len());

        let mut entities: Vec<serde_json::Value> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for uid in &all_uids {
            if seen.contains(uid) { continue; }
            seen.insert(uid.clone());
            if let Ok(raw) = tx.load(uid) {
                let class = raw.value["class"].as_str().unwrap_or("?");
                if class == "IndexedEntity" { continue; }
                match render_entity(&render_instance, &mut bp, tx, &raw.value, true) {
                    Ok(r) => { entities.push(r); }
                    Err(e) => { eprintln!("[hexroll3-cli] WARN render {}: {}", uid, e); }
                }
            }
        }
        Ok(entities)
    })?;

    let json = serde_json::json!({
        "version": 1, "sandbox_id": sid,
        "entity_count": exported.len(), "entities": exported,
    });
    std::fs::write(output, serde_json::to_string_pretty(&json)?)?;
    eprintln!("[hexroll3-cli] Wrote {} entities → {:?}", exported.len(), output);
    eprintln!("[hexroll3-cli] Done ✓");
    Ok(())
}
