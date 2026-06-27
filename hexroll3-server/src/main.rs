//! hexroll3-server — HTTP wrapper around the hexroll3 engine.
//!
//! Exposes a single endpoint the Hexxkraw frontend queries to generate
//! seeded OSR worlds on demand:
//!
//!   POST /generate   { "seed": 42 }            → full world JSON
//!   GET  /health                               → { "ok": true }
//!
//! Generation is reproducible: the same seed always produces the same world.
//! CORS is permissive so the Vite dev server (localhost:5173) can call it.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use clap::Parser;
use hexroll3_scroll::{
    generators::append,
    instance::{SandboxBuilder, SandboxInstance},
    renderer::render_entity,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tower_http::cors::{Any, CorsLayer};

#[derive(Parser, Clone)]
#[command(name = "hexroll3-server")]
struct Cli {
    /// Directory containing the processed scroll files (must hold main.scroll).
    #[arg(long, default_value = "/tmp/hexroll-osr-scrolls-processed")]
    scroll_dir: PathBuf,

    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1:8787")]
    bind: String,

    // ── Internal worker mode ────────────────────────────────────────────────
    // Each /generate request re-execs this same binary with `--worker` to do the
    // actual (CPU-heavy, sometimes pathological) world roll in a *separate
    // process*. The parent waits with a deadline and, on overrun, SIGKILLs the
    // child — which actually frees the CPU. Doing the roll on an in-process
    // thread could not be cancelled (a looping seed pinned a core forever and
    // poisoned every later request); a child process can.
    /// Internal: generate one world, write JSON to --out, exit. Not for direct use.
    #[arg(long, hide = true)]
    worker: bool,
    /// Worker: optional seed for reproducible worlds.
    #[arg(long, hide = true)]
    seed: Option<u64>,
    /// Worker: map size ("small"/"medium"/"large"/"giant").
    #[arg(long, hide = true)]
    map_size: Option<String>,
    /// Worker: path to write the generated world JSON to.
    #[arg(long, hide = true)]
    out: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    scroll_dir: Arc<PathBuf>,
    /// Bounds how many generation subprocesses run at once so a burst of
    /// requests can't oversubscribe the CPU (each worker is single-threaded but
    /// CPU-bound). Sized to leave a core for the async runtime + I/O.
    gen_slots: Arc<tokio::sync::Semaphore>,
}

/// Monotonic counter to keep concurrent sandbox temp files unique even for
/// identical seeds (the on-disk redb file must not be shared between rolls).
static SANDBOX_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Deserialize)]
struct GenerateRequest {
    /// Optional seed for reproducible worlds. Omit for a random world.
    seed: Option<u64>,
    /// Map size requested by the game's size chip: "small" / "medium" /
    /// "large" / "giant". Bounds the realm (regions × tiles per region) so
    /// small maps generate fast and rarely time out. Omit → "medium".
    map_size: Option<String>,
}

/// Realm sizing per map-size chip, chosen so total hexes ≈ the game's targets
/// (small 91 · medium 169 · large 331 · giant 631). Region count is fixed
/// (min == max) for predictable size; tiles-per-region is a tight band.
///
/// Cost model (measured, not assumed): the initial realm `create()` roll
/// dominates — it instantiates every hex (terrain + roaming monster) and scales
/// ~linearly with hex count (~95 hexes ≈ 9s, ~165 ≈ 19s). Dungeon excavation is
/// cheap (~5 dungeons in <1s), so `max_dungeons` is a minor knob and mainly
/// bounds payload size. Pathological seeds (a settlement/dungeon instance that
/// expands without bound) are handled by killing the worker process, not by
/// sizing — so the timeouts only need headroom over the *legitimate* roll cost.
#[derive(Clone, Copy)]
struct RealmSizing {
    regions: i64,
    tiles_min: i64,
    tiles_max: i64,
    /// Dungeons excavated across the realm. Excavation is cheap; this mostly
    /// caps payload size and dungeon density.
    max_dungeons: usize,
    /// Wilderness features (landmarks: watchtower, graveyard, arena…) appended
    /// onto hexes. Cheap to roll; gives the overworld named POIs beyond
    /// settlements/dungeons. Bounded so payload/gen-cost stay predictable.
    max_features: usize,
    /// Wall-clock budget before the worker is killed and the client falls back.
    timeout_secs: u64,
}

/// Reads `HEXROLL_TIMEOUT_SCALE` from the environment (default 1.0). Values
/// above 1.0 multiply every per-size timeout proportionally — useful when the
/// host machine is slower than the benchmarked baseline (e.g. Docker on ARM,
/// constrained CI). Set to 3 in docker-compose for the typical dev environment.
fn timeout_scale() -> f64 {
    std::env::var("HEXROLL_TIMEOUT_SCALE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .unwrap_or(1.0)
}

fn realm_sizing(map_size: Option<&str>) -> RealmSizing {
    // Base timeouts ≈ measured roll cost (~0.1s/hex) + generous margin. The
    // margin is safe to keep wide because an overrunning worker is SIGKILLed
    // (it cannot leak CPU into later requests), so a high ceiling never
    // poisons the server. Multiply by HEXROLL_TIMEOUT_SCALE for slow hosts.
    let scale = timeout_scale();
    let t = |base: u64| ((base as f64) * scale).ceil() as u64;
    match map_size.unwrap_or("medium") {
        // ~5×19 ≈ 95 hexes · ~10s (base)
        "small" => RealmSizing { regions: 5, tiles_min: 16, tiles_max: 22, max_dungeons: 5, max_features: 6, timeout_secs: t(45) },
        // ~12×28 ≈ 336 hexes · ~40s (base)
        "large" => RealmSizing { regions: 12, tiles_min: 24, tiles_max: 32, max_dungeons: 15, max_features: 18, timeout_secs: t(115) },
        // ~19×33 ≈ 627 hexes · ~75s (base)
        "giant" => RealmSizing { regions: 19, tiles_min: 30, tiles_max: 36, max_dungeons: 22, max_features: 28, timeout_secs: t(165) },
        // medium ~8×21 ≈ 168 hexes · ~20s (base)
        _ => RealmSizing { regions: 8, tiles_min: 18, tiles_max: 24, max_dungeons: 9, max_features: 10, timeout_secs: t(70) },
    }
}

/// Lean, game-focused world description. The frontend lays out the navigable
/// grid from `regions`/`hexes` (terrain + encounters), places `settlements`
/// and `dungeons` by their `hex` link, and renders dungeon interiors from
/// `areas`. Replaces the old "dump every entity" payload (which ballooned to
/// 11 MB once settlements/dungeons existed).
#[derive(Serialize)]
struct GenerateResponse {
    version: u32,
    seed: Option<u64>,
    sandbox_id: String,
    realm: RealmInfo,
    regions: Vec<RegionInfo>,
    settlements: Vec<SettlementInfo>,
    dungeons: Vec<DungeonInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    factions: Vec<FactionBrief>,
}

#[derive(Serialize)]
struct FactionBrief {
    /// Nome completo, ex.: "The Bloodied Veil".
    name: String,
    /// "Cult" / "Militia" / "Syndicate".
    kind: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    leader: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    alignment: String,
    /// uid da masmorra-covil (FactionLair.DungeonUUID), se houver.
    #[serde(skip_serializing_if = "String::is_empty")]
    lair_dungeon: String,
}

#[derive(Serialize)]
struct RealmInfo {
    name: String,
    ruler: String,
    background: String,
    realm_type: String,
}

#[derive(Serialize)]
struct RegionInfo {
    name: String,
    /// "Plains" / "Jungle" / "Swamps" / "Tundra" / "Forest" / "Mountains" / …
    terrain: String,
    hexes: Vec<HexInfo>,
}

#[derive(Serialize)]
struct HexInfo {
    uid: String,
    terrain: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encounter: Option<MonsterBrief>,
    /// uid of a settlement on this hex (see top-level `settlements`).
    #[serde(skip_serializing_if = "Option::is_none")]
    settlement: Option<String>,
    /// uid of a dungeon on this hex (see top-level `dungeons`).
    #[serde(skip_serializing_if = "Option::is_none")]
    dungeon: Option<String>,
    /// Wilderness landmark on this hex (watchtower, graveyard, arena…).
    #[serde(skip_serializing_if = "Option::is_none")]
    feature: Option<FeatureBrief>,
}

#[derive(Serialize, Clone)]
struct FeatureBrief {
    name: String,
    description: String,
}

#[derive(Serialize, Clone)]
struct MonsterBrief {
    name: String,
    hit_dice: String,
    armour_class: String,
    attacks: String,
    thac0: String,
    movement: String,
    morale: String,
    saving_throws: String,
    alignment: String,
    xp: String,
    number_appearing: String,
    treasure_type: String,
}

#[derive(Serialize)]
struct SettlementInfo {
    uid: String,
    name: String,
    /// "Town" / "Village" / "Hamlet" / "City" …
    kind: String,
    region: String,
    hex: String,
    population: String,
    /// NPCs notáveis do assentamento (candidatos a hireling).
    npcs: Vec<NpcBrief>,
    /// Ganchos de aventura / missões oferecidos no assentamento.
    quests: Vec<QuestBrief>,
    /// Taverna/estalagem do assentamento (nome, tipo, prato típico).
    #[serde(skip_serializing_if = "Option::is_none")]
    tavern: Option<TavernBrief>,
    /// Lojas tipadas do assentamento (nome + tipo).
    shops: Vec<ShopBrief>,
}

#[derive(Serialize)]
struct QuestBrief {
    /// "treasure" / "missing-person" / "escort" / "delivery"
    kind: String,
    /// Texto da missão já renderizado pelo motor (nomes, recompensa, local).
    text: String,
}

#[derive(Serialize)]
struct TavernBrief {
    /// Nome próprio da taverna, ex.: "The Sad Goblin Tavern".
    name: String,
    /// Tipo: "Tavern" / "Lodge" / "Inn" …
    kind: String,
    /// Prato típico (flavor), se disponível.
    dish: String,
}

#[derive(Serialize)]
struct ShopBrief {
    /// Nome próprio, ex.: "Aqualina's Weeds".
    name: String,
    /// Tipo de loja, ex.: "Herbalist" / "Fish Market".
    kind: String,
    /// Multiplicador de preço do distrito (hexroll CostFactor; 1.0 = padrão).
    #[serde(skip_serializing_if = "is_default_cost")]
    cost_factor: f64,
}

fn is_default_cost(c: &f64) -> bool {
    (*c - 1.0).abs() < f64::EPSILON
}

#[derive(Serialize)]
struct NpcBrief {
    name: String,
    /// classe base em minúsculas: fighter/cleric/magicuser/thief/dwarf/elf/halfling
    class: String,
    level: i64,
    hp: String,
    armour_class: String,
    thac0: String,
    alignment: String,
}

#[derive(Serialize)]
struct DungeonInfo {
    uid: String,
    name: String,
    /// "Temple" / "Tomb" / "Cavern" …
    kind: String,
    region: String,
    hex: String,
    entrances: String,
    areas: Vec<AreaInfo>,
    /// Wandering monster table (DungeonWanderingMonsters.monsters), deduped by name.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    wandering: Vec<MonsterBrief>,
}

#[derive(Serialize)]
struct AreaInfo {
    number: i64,
    x: i64,
    y: i64,
    title: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encounter: Option<MonsterBrief>,
    /// Ouro plantado na sala (derivado do tier — o valor exato do hexroll é um
    /// template não-resolvido headless). 0 = sem tesouro.
    treasure_gold: i64,
    /// Nomes de itens mágicos encontrados na sala (se houver).
    treasure_items: Vec<String>,
    /// Armadilha da sala (de Feature.AreaTrap), se houver.
    #[serde(skip_serializing_if = "Option::is_none")]
    trap: Option<TrapBrief>,
    /// Números das salas conectadas por passagem (grafo real de corredores).
    connections: Vec<i64>,
    /// Subset of `connections` reached only via a secret door (hidden until found).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    secret: Vec<i64>,
}

#[derive(Serialize)]
struct TrapBrief {
    name: String,
    /// Prosa descrevendo a armadilha (inclui save/dano; o front extrai).
    description: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let cli = Cli::parse();

    let main_scroll = cli.scroll_dir.join("main.scroll");
    if !main_scroll.exists() {
        return Err(anyhow!(
            "main.scroll not found in {:?}. Preprocess the scrolls first.",
            cli.scroll_dir
        ));
    }

    // Worker mode: do one generation in this (disposable, killable) process and
    // exit. The parent server spawns us per request and kills us on timeout.
    if cli.worker {
        return run_worker(&cli);
    }

    // Allow as many concurrent generation subprocesses as we have spare cores
    // (keep one for the runtime). Generation is CPU-bound and single-threaded
    // per worker, so this maps ~1 worker per core without thrashing.
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let gen_slots = Arc::new(tokio::sync::Semaphore::new(cores.saturating_sub(1).max(1)));

    let state = AppState {
        scroll_dir: Arc::new(cli.scroll_dir.clone()),
        gen_slots,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&cli.bind).await?;
    tracing::info!("hexroll3-server listening on http://{}", cli.bind);
    tracing::info!("scroll dir: {:?}", cli.scroll_dir);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "ok": true }))
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    let seed = req.seed;
    let map_size = req.map_size.clone();
    let sizing = realm_sizing(map_size.as_deref());
    let timeout_secs = sizing.timeout_secs;

    // The roll is CPU-heavy and, for some seeds, *pathological*: certain
    // settlement/dungeon instances expand into an enormous (or genuinely
    // infinite) entity subtree that pins a core. We cannot fix those data cases
    // from outside, and an in-process worker thread cannot be cancelled — a
    // timed-out roll would loop forever, burning a core and poisoning every
    // later request (a cascade where, after a few bad seeds, nearly everything
    // times out). So we run each generation in a **separate child process** and
    // SIGKILL it on overrun, which truly reclaims the CPU. The client gets a
    // fast 500 and falls back to offline gen; the next request is unaffected.
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("current_exe failed: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "server misconfigured" })),
            )
                .into_response();
        }
    };

    // Unique output path per request (pid + counter): the worker writes the
    // world JSON here and we stream it back, then delete it.
    let n = SANDBOX_COUNTER.fetch_add(1, Ordering::Relaxed);
    let out_path = std::env::temp_dir().join(format!(
        "hexroll3-world-{}-{}-{}.json",
        std::process::id(),
        seed.unwrap_or(0),
        n
    ));

    // Cap concurrent generations so a burst can't oversubscribe the CPU.
    let _permit = state.gen_slots.acquire().await.expect("semaphore open");

    let mut cmd = tokio::process::Command::new(&exe);
    cmd.arg("--worker")
        .arg("--scroll-dir")
        .arg(&*state.scroll_dir)
        .arg("--out")
        .arg(&out_path)
        .kill_on_drop(true);
    if let Some(s) = seed {
        cmd.arg("--seed").arg(s.to_string());
    }
    if let Some(ms) = &map_size {
        cmd.arg("--map-size").arg(ms);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("failed to spawn generation worker: {e}");
            let _ = std::fs::remove_file(&out_path);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "could not start generation" })),
            )
                .into_response();
        }
    };

    let status = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        child.wait(),
    )
    .await;

    let response = match status {
        // Worker exited in time and succeeded → stream its JSON back verbatim.
        Ok(Ok(st)) if st.success() => match tokio::fs::read(&out_path).await {
            Ok(bytes) => (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                bytes,
            )
                .into_response(),
            Err(e) => {
                tracing::error!("worker ok but output unreadable: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "generation output missing" })),
                )
                    .into_response()
            }
        },
        // Worker ran but failed (e.g. roll error) → 500, client falls back.
        Ok(Ok(st)) => {
            tracing::error!("generation worker exited with {st} (seed {seed:?}, size {map_size:?})");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "generation failed" })),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            tracing::error!("waiting on generation worker failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "generation failed" })),
            )
                .into_response()
        }
        // Deadline hit → kill the (looping) child so it stops eating CPU.
        Err(_) => {
            tracing::warn!(
                "generation timed out after {timeout_secs}s (seed {seed:?}, size {map_size:?}); killing worker"
            );
            let _ = child.kill().await;
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "generation timed out" })),
            )
                .into_response()
        }
    };

    let _ = tokio::fs::remove_file(&out_path).await;
    response
}

/// Worker entry point: generate one world and write it as JSON to `--out`.
/// Runs in a disposable child process the server kills on timeout.
fn run_worker(cli: &Cli) -> Result<()> {
    let out = cli
        .out
        .clone()
        .ok_or_else(|| anyhow!("--worker requires --out"))?;
    let sizing = realm_sizing(cli.map_size.as_deref());
    let resp = generate_world(&cli.scroll_dir, cli.seed, sizing)?;
    let json = serde_json::to_vec(&resp)?;
    std::fs::write(&out, json)?;
    Ok(())
}

/// Generate a fresh world, render every entity, and return it in-memory.
fn generate_world(scroll_dir: &PathBuf, seed: Option<u64>, sizing: RealmSizing) -> Result<GenerateResponse> {
    let n = SANDBOX_COUNTER.fetch_add(1, Ordering::Relaxed);
    // Include the pid: workers are separate processes (their counters all start
    // at 0), so two concurrent workers with the same seed would otherwise share
    // a redb path.
    let sandbox_path = std::env::temp_dir().join(format!(
        "hexroll3-server-{}-{}-{}.hxr",
        std::process::id(),
        seed.unwrap_or(0),
        n
    ));
    // Fresh sandbox every time.
    let _ = std::fs::remove_file(&sandbox_path);

    let mut instance = SandboxInstance::new();
    if let Some(s) = seed {
        instance.with_seed(s);
    }
    let t0 = std::time::Instant::now();
    instance.with_scroll(scroll_dir.join("main.scroll"))?;
    tracing::info!("with_scroll: {:.1}s", t0.elapsed().as_secs_f32());
    // Wire the cartographer's data provider: it fills the OSR scrolls' empty
    // `DungeonMap {}` / `CaveMap {}` placeholders with a real excavated interior
    // — rooms (with coordinates), doors, secret doors, passages, and per-area
    // RoomDescriptions. The OSR scrolls define the RoomType subtypes it rolls
    // (we add the one it omits, RoomTypeThrone, in preprocess-scrolls.py).
    {
        let mut bp = instance
            .blueprint
            .lock()
            .map_err(|_| anyhow!("blueprint lock"))?;
        bp.map_data_provider = hexroll3_cartographer::dungeons::map_data_providers();

        // Override the realm-size globals parsed from the scroll so the world
        // matches the requested map-size chip. Region count is fixed and the
        // tiles-per-region band is tightened; this is read at create() time,
        // so setting it after with_scroll() (which loaded the defaults) wins.
        //
        // CRITICAL: also zero out settlements, dungeons and factions so that
        // create() only builds the geographic skeleton (regions + hexes +
        // terrain + roaming encounters). The scroll defaults for these are
        // 6-9 dungeons, 6-9 settlements and 3-5 factions, which means the
        // cartographer would excavate 6-9 extra dungeons during create() ON
        // TOP of what populate_features() adds — causing most seeds to exceed
        // the timeout. populate_features() is the sole path for all content.
        bp.globals.insert("minimum_number_of_regions".into(), serde_json::json!(sizing.regions));
        bp.globals.insert("maximum_number_of_regions".into(), serde_json::json!(sizing.regions));
        bp.globals.insert("minimum_number_of_dungeons".into(), serde_json::json!(0));
        bp.globals.insert("maximum_number_of_dungeons".into(), serde_json::json!(0));
        bp.globals.insert("minimum_number_of_settlements".into(), serde_json::json!(0));
        bp.globals.insert("maximum_number_of_settlements".into(), serde_json::json!(0));
        bp.globals.insert("minimum_number_of_factions".into(), serde_json::json!(0));
        bp.globals.insert("maximum_number_of_factions".into(), serde_json::json!(0));
        // Note: settlement class dispatch uses a hardcoded `settlement_classes`
        // list (Village:3, Town:2, City:2) that cannot be overridden via globals.
        // We force Village via the class name in append() calls instead.
        for terr in ["mountains", "forest", "desert", "plains", "jungle", "swamps", "tundra"] {
            bp.globals.insert(format!("minimum_tiles_per_{terr}_region"), serde_json::json!(sizing.tiles_min));
            bp.globals.insert(format!("maximum_tiles_per_{terr}_region"), serde_json::json!(sizing.tiles_max));
        }
    }
    let t1 = std::time::Instant::now();
    instance.create(
        sandbox_path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 sandbox path"))?,
    )?;
    tracing::info!("create: {:.1}s", t1.elapsed().as_secs_f32());

    // Drive hexroll's own generators headless to populate settlements & dungeons
    // (the initial roll only fills terrain + a roaming monster per hex; features
    // are normally rolled on-demand when a user clicks a hex in the app).
    let t2 = std::time::Instant::now();
    populate_features(&instance, sizing)?;
    tracing::info!("populate_features: {:.1}s", t2.elapsed().as_secs_f32());

    let t3 = std::time::Instant::now();
    let resp = export_all(instance, seed)?;
    tracing::info!("export_all: {:.1}s", t3.elapsed().as_secs_f32());

    // Best-effort cleanup; the .hxr is a transient artifact.
    let _ = std::fs::remove_file(&sandbox_path);
    Ok(resp)
}

/// Collect every hex UID, grouped by region, from the realm tree.
fn hexes_by_region(instance: &SandboxInstance) -> Result<Vec<Vec<String>>> {
    instance.repo.inspect(|tx| {
        let mut out: Vec<Vec<String>> = Vec::new();
        // "root" holds the main entity uid; main has realms[], each realm has
        // regions[], each region has a Hexmap[] of hex uids.
        let root = tx.load("root")?;
        let main_uid = root
            .value
            .as_str()
            .ok_or_else(|| anyhow!("root not string"))?
            .to_string();
        let main = tx.load(&main_uid)?;
        let Some(realms) = main.value.get("realms").and_then(|v| v.as_array())
        else {
            return Ok(out);
        };
        for realm_uid in realms.iter().filter_map(|v| v.as_str()) {
            let Ok(realm) = tx.load(realm_uid) else { continue };
            let Some(regions) =
                realm.value.get("regions").and_then(|v| v.as_array())
            else {
                continue;
            };
            for region_uid in regions.iter().filter_map(|v| v.as_str()) {
                let Ok(region) = tx.load(region_uid) else { continue };
                let Some(hexes) =
                    region.value.get("Hexmap").and_then(|v| v.as_array())
                else {
                    continue;
                };
                let uids: Vec<String> = hexes
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                if !uids.is_empty() {
                    out.push(uids);
                }
            }
        }
        Ok(out)
    })
}

/// Roll settlements & dungeons onto a seed-chosen subset of hexes by invoking
/// hexroll's `append` generator — the same path the interactive app uses. With
/// the cartographer provider wired, dungeon appends also generate interiors.
fn populate_features(instance: &SandboxInstance, sizing: RealmSizing) -> Result<()> {
    let regions = hexes_by_region(instance)?;
    if regions.is_empty() {
        return Ok(());
    }

    let render_instance = instance.clone();
    let builder = SandboxBuilder::from_instance(&render_instance);
    let mut blueprint = builder
        .sandbox
        .blueprint
        .lock()
        .map_err(|_| anyhow!("blueprint lock"))?;

    tracing::info!(
        "populate: {} regions, {} hexes total",
        regions.len(),
        regions.iter().map(|r| r.len()).sum::<usize>()
    );
    // One settlement per region; dungeons are budgeted (`sizing.max_dungeons`)
    // and spread round-robin across regions. Dungeon excavation dominates
    // generation time, so bounding the count is what keeps small maps fast and
    // off the timeout. We do NOT force settlement/dungeon *classes* — that just
    // shifts the RNG stream and reshuffles which seeds hit data-dependent
    // scroll loops; the wall-clock timeout still bounds those.
    let region_count = regions.len();
    builder.sandbox.repo.mutate(|tx| {
        let mut settle_idxs: Vec<usize> = Vec::with_capacity(region_count);
        // Hexes already carrying a settlement/dungeon — features avoid them so a
        // landmark never collides with a town/dungeon on the same tile.
        let mut used_hexes: std::collections::HashSet<String> = std::collections::HashSet::new();
        for hexes in &regions {
            let n = hexes.len();
            if n == 0 {
                settle_idxs.push(usize::MAX);
                continue;
            }
            let settle_idx = builder.randomizer.in_range(0, n as i32 - 1) as usize;
            settle_idxs.push(settle_idx);
            // Force Village class_override: the generic "Settlement" dispatch
            // can pick City/Town subtypes whose deeply nested NPC sub-trees
            // enter data-dependent infinite loops for many seeds. Village is
            // structurally simpler and avoids most of those cases.
            if let Err(e) =
                append(&builder, &mut blueprint, tx, &hexes[settle_idx], "Settlement", Some("Village"), 1)
            {
                tracing::warn!("append Settlement: {e:#}");
            } else {
                used_hexes.insert(hexes[settle_idx].clone());
            }
        }

        // Spread the dungeon budget across regions (round-robin) so coverage is
        // even rather than front-loading the first regions.
        let mut placed = 0usize;
        let mut round = 0usize;
        while placed < sizing.max_dungeons {
            let mut any = false;
            for (ri, hexes) in regions.iter().enumerate() {
                if placed >= sizing.max_dungeons {
                    break;
                }
                let n = hexes.len();
                if n == 0 || round >= n {
                    continue;
                }
                any = true;
                let di = builder.randomizer.in_range(0, n as i32 - 1) as usize;
                if di == *settle_idxs.get(ri).unwrap_or(&usize::MAX) {
                    continue;
                }
                if let Err(e) =
                    append(&builder, &mut blueprint, tx, &hexes[di], "Dungeon", None, 1)
                {
                    tracing::warn!("append Dungeon: {e}");
                } else {
                    used_hexes.insert(hexes[di].clone());
                }
                placed += 1;
            }
            if !any {
                break; // every region exhausted
            }
            round += 1;
        }

        // Wilderness features (landmarks) — spread round-robin across regions,
        // one per free hex, up to the budget. Skips hexes already used by a
        // settlement/dungeon so a landmark never shares a tile.
        let mut feats = 0usize;
        let mut frnd = 0usize;
        while feats < sizing.max_features {
            let mut any = false;
            for hexes in regions.iter() {
                if feats >= sizing.max_features { break; }
                let n = hexes.len();
                if n == 0 || frnd >= n { continue; }
                any = true;
                let fi = builder.randomizer.in_range(0, n as i32 - 1) as usize;
                let hex = &hexes[fi];
                if used_hexes.contains(hex) { continue; }
                if let Err(e) = append(&builder, &mut blueprint, tx, hex, "Feature", None, 1) {
                    tracing::warn!("append Feature: {e}");
                } else {
                    used_hexes.insert(hex.clone());
                    feats += 1;
                }
            }
            if !any { break; }
            frnd += 1;
        }

        // Facções (cultos/milícias/sindicatos): geradas via append no realm (o
        // scroll declara [0..0 factions], então não saem do create() inicial).
        // Cada uma traz líder e covil (FactionLair → dungeon).
        let mut faction_count = 0usize;
        let main_uid = tx.load("root").ok()
            .and_then(|root| root.as_str().map(str::to_string));
        let realm_uid = main_uid.as_deref()
            .and_then(|uid| tx.load(uid).ok())
            .and_then(|main| {
                main["realms"].as_array()
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            });
        if let Some(realm_uid) = realm_uid {
            let n = builder.randomizer.in_range(3, 5) as u32;
            match append(&builder, &mut blueprint, tx, &realm_uid, "factions", None, n) {
                Ok(uids) => faction_count = uids.len(),
                Err(e) => { tracing::warn!("append Faction: {e}"); }
            }
        }
        tracing::info!("populate: {} settlements, {} dungeons (budget {}), {} features (budget {}), {} factions", region_count, placed, sizing.max_dungeons, feats, sizing.max_features, faction_count);
        Ok(())
    })?;
    Ok(())
}

// ── Field extraction helpers ────────────────────────────────────────────────

/// Strip HTML tags, unresolved minijinja fragments (`{{…}}` / `{%…%}`), and
/// collapse whitespace from a rendered template string.
fn html_to_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut in_tag = false;
    while i < bytes.len() {
        let c = bytes[i] as char;
        // Skip {{ ... }} and {% ... %} jinja fragments left unresolved.
        if c == '{' && i + 1 < bytes.len() && (bytes[i + 1] == b'{' || bytes[i + 1] == b'%') {
            let close: &[u8] = if bytes[i + 1] == b'{' { b"}}" } else { b"%}" };
            if let Some(end) = s[i..].find(std::str::from_utf8(close).unwrap()) {
                i += end + 2;
                continue;
            }
        }
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Recursively search a rendered value for the first monster-like object
/// (has Title + HitDice + THAC0). Dungeon-room encounters are nested inside
/// the room's `Feature` (FeatureLevelClass → encounter → monster), so a direct
/// field lookup misses them.
fn find_monster(v: &Value) -> Option<MonsterBrief> {
    match v {
        Value::Object(o) => {
            if o.contains_key("HitDice") && o.contains_key("THAC0") && o.contains_key("Title") {
                if let Some(m) = monster_brief(v) {
                    return Some(m);
                }
            }
            for (_, child) in o {
                if let Some(m) = find_monster(child) {
                    return Some(m);
                }
            }
            None
        }
        Value::Array(a) => a.iter().find_map(find_monster),
        _ => None,
    }
}

/// Read a field as a clean string, unwrapping the common `{Title: "..."}` /
/// `{Full: "..."}` / `{Name: "..."}` nested shapes and stripping HTML.
fn field_str(v: &Value, key: &str) -> String {
    match &v[key] {
        Value::String(s) => html_to_text(s),
        Value::Number(n) => n.to_string(),
        Value::Object(o) => {
            for k in ["Title", "Full", "Name"] {
                if let Some(Value::String(s)) = o.get(k) {
                    return html_to_text(s);
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

/// Build a MonsterBrief from a rendered Monster object, if present.
fn monster_brief(v: &Value) -> Option<MonsterBrief> {
    if !v.is_object() {
        return None;
    }
    let name = field_str(v, "Title");
    if name.is_empty() {
        return None;
    }
    let na_roam = field_str(v, "NumberAppearingRoaming");
    let na_lair = field_str(v, "NumberAppearingLair");
    let number_appearing = if !na_roam.is_empty() {
        na_roam
    } else {
        na_lair
    };
    let treasure_type = match &v["TreasureType"] {
        Value::Object(o) => o
            .get("class")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .trim_start_matches("TreasureType")
            .to_string(),
        Value::String(s) => s.clone(),
        _ => String::new(),
    };
    Some(MonsterBrief {
        name,
        hit_dice: field_str(v, "HitDice"),
        armour_class: field_str(v, "ArmourClass"),
        attacks: field_str(v, "Attacks"),
        thac0: field_str(v, "THAC0"),
        movement: field_str(v, "Movement"),
        morale: field_str(v, "Morale"),
        saving_throws: field_str(v, "SavingThrows"),
        alignment: field_str(v, "Alignment"),
        xp: field_str(v, "XP"),
        number_appearing,
        treasure_type,
    })
}

/// "SwingingBladeTrap" → "Swinging Blade Trap".
fn humanize_class(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && c.is_uppercase() {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

/// Deterministic FNV-1a hash of a string (for seed-stable derived values).
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Recursively collect magic-item names (objects whose class contains
/// "MagicItem") from a rendered value.
fn find_magic_items(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(o) => {
            if o.get("class").and_then(|c| c.as_str()).is_some_and(|c| c.contains("MagicItem")) {
                let name = field_str(v, "Title");
                let name = if name.is_empty() { field_str(v, "Name") } else { name };
                if !name.is_empty() && !out.contains(&name) {
                    out.push(name);
                }
            }
            for (_, child) in o {
                find_magic_items(child, out);
            }
        }
        Value::Array(a) => a.iter().for_each(|x| find_magic_items(x, out)),
        _ => {}
    }
}

/// Parse a hexroll NPC class like "FighterLevel5" → ("fighter", 5).
fn npc_class_level(class: &str) -> Option<(String, i64)> {
    for base in ["Fighter", "Cleric", "Magicuser", "Thief", "Dwarf", "Elf", "Halfling"] {
        if let Some(rest) = class.strip_prefix(base) {
            if let Some(num) = rest.strip_prefix("Level") {
                if let Ok(lvl) = num.parse::<i64>() {
                    return Some((base.to_lowercase(), lvl));
                }
            }
        }
    }
    None
}

/// Normaliza prosa renderizada headless: desfaz mojibake (UTF-8 reinterpretado
/// como latin1, possivelmente em dois níveis) e colapsa espaços. Seguro: se a
/// reinterpretação não for UTF-8 válido (ex.: acento legítimo), mantém o texto.
fn clean_prose(s: &str) -> String {
    let mut t = s.to_string();
    for _ in 0..2 {
        if t.chars().all(|c| (c as u32) <= 0xFF) {
            let bytes: Vec<u8> = t.chars().map(|c| c as u8).collect();
            match String::from_utf8(bytes) {
                Ok(u) if u != t => { t = u; }
                _ => break,
            }
        } else {
            break;
        }
    }
    // Resíduos de nível único (€/™/aspas curvas fora do alcance latin1).
    t = t.replace("â€™", "'").replace("â€œ", "\"").replace("â€", "\"");
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Remove aspas (retas ou curvas) ao redor de um nome.
fn strip_quotes(s: &str) -> String {
    s.trim()
        .trim_matches(|c| c == '"' || c == '\'' || c == '“' || c == '”')
        .trim()
        .to_string()
}

/// Mapeia a classe de uma entidade de quest do hexroll para um tipo curto.
/// Retorna `None` para classes que não são missões oferecíveis.
fn quest_kind(class: &str) -> Option<String> {
    match class {
        "TreasureQuest" => Some("treasure".to_string()),
        "MissingPersonQuest" => Some("missing-person".to_string()),
        "EscortQuest" => Some("escort".to_string()),
        "DeliveryQuest" => Some("delivery".to_string()),
        _ => None,
    }
}

/// Limpa o texto renderizado de uma quest: colapsa espaços, conserta mojibake
/// de apóstrofo e remove artefatos de links não resolvidos headless
/// ("held captive in the)" → "held captive somewhere.", "(hex )"…).
fn clean_quest_text(s: &str) -> String {
    let mut t = s.split_whitespace().collect::<Vec<_>>().join(" ");
    // Apóstrofo curvo (U+2019) decodificado errado vira "â"; normaliza.
    t = t.replace("â€™", "'").replace('â', "'");
    t = t
        .replace("(hex )", "")
        .replace("in the)", "somewhere.")
        .replace("in the )", "somewhere.")
        .replace(" in and ", " and ")
        .replace(" in the Finder", ". The finder")
        .replace(" in .", ".")
        .replace(" )", "")
        .replace("( ", "(")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace("..", ".");
    while t.contains("  ") {
        t = t.replace("  ", " ");
    }
    t.trim().to_string()
}

/// Uma quest renderizada é boa o bastante para mostrar? Descarta as que
/// dependem de links vazios headless (entrega sem destino/recompensa).
fn quest_text_ok(t: &str) -> bool {
    let low = t.to_lowercase();
    let words = t.split_whitespace().count();
    words >= 6
        && !low.ends_with("reward is")
        && !low.ends_with("reward is.")
        && !low.contains("deliver to. reward")
}

/// Recursively collect UID-looking strings from a JSON value (8-char
/// alphanumerics), following UUID-keyed references to a bounded depth.
fn collect_uids_recursive(val: &Value, uids: &mut Vec<String>, depth: u32) {
    if depth == 0 {
        return;
    }
    match val {
        Value::String(s) if s.len() == 8 => {
            if s.chars().all(|c| c.is_alphanumeric()) {
                uids.push(s.clone());
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_uids_recursive(item, uids, depth - 1);
            }
        }
        Value::Object(map) => {
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

/// Extract a lean, game-focused world description: realm identity, regions with
/// their terrain hexes + encounters, settlements (summary), and dungeons with
/// full interiors (rooms, coords, encounters, connections).
fn export_all(instance: SandboxInstance, seed: Option<u64>) -> Result<GenerateResponse> {
    let sid = instance.sid().ok_or_else(|| anyhow!("No sandbox ID"))?;
    let render_instance = instance.clone(); // shares Arc<Mutex<blueprint>>

    instance.repo.inspect(|tx| {
        let mut bp = render_instance
            .blueprint
            .lock()
            .map_err(|_| anyhow!("blueprint lock"))?;

        // Render the raw stored value of a uid into its template form.
        macro_rules! render_uid {
            ($uid:expr) => {
                tx.load($uid).ok().and_then(|raw| {
                    render_entity(&render_instance, &mut bp, tx, &raw.value, true).ok()
                })
            };
        }

        // root → main entity → realms[]
        let main_uid = tx
            .load("root")?
            .value
            .as_str()
            .ok_or_else(|| anyhow!("root not string"))?
            .to_string();
        let main_raw = tx.load(&main_uid)?;
        let realm_uids: Vec<String> = main_raw.value["realms"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        let realm_uid = realm_uids
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("no realm"))?;

        // ── Realm identity ──────────────────────────────────────────────
        let realm_r = render_uid!(&realm_uid).unwrap_or(Value::Null);
        let realm = RealmInfo {
            name: field_str(&realm_r, "Title"),
            ruler: field_str(&realm_r, "RulerTitle")
                .replace(" of none", "")
                .trim()
                .to_string(),
            background: field_str(&realm_r, "Background"),
            realm_type: field_str(&realm_r, "RealmType"),
        };

        // ── Factions (cults/militias/syndicates) + lairs ────────────────
        let faction_uids: Vec<String> = tx.load(&realm_uid)?.value["factions"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        let mut factions: Vec<FactionBrief> = Vec::new();
        for fu in &faction_uids {
            let kind = tx.load(fu).ok()
                .and_then(|raw| raw.value["class"].as_str().map(str::to_string))
                .unwrap_or_default();
            let Some(f) = render_uid!(fu) else { continue };
            // FactionName desdobra para {Full|Title}; idem leader e lair.
            let name = {
                let n = field_str(&f["FactionName"], "Full");
                if n.is_empty() { field_str(&f, "FactionName") } else { n }
            };
            if name.is_empty() { continue; }
            let leader = {
                let l = field_str(&f["FactionLeader"]["Name"], "Full");
                if l.is_empty() { field_str(&f["FactionLeader"], "Name") } else { l }
            };
            let alignment = {
                let a = field_str(&f, "Alignment");
                if a.is_empty() { field_str(&f, "AcceptedAlignment") } else { a }
            };
            let lair_dungeon = {
                let d = field_str(&f["FactionLair"], "DungeonUUID");
                if d.is_empty() { field_str(&f["Lair"], "DungeonUUID") } else { d }
            };
            factions.push(FactionBrief {
                name: clean_prose(&name), kind: humanize_class(&kind),
                leader: clean_prose(&leader), alignment, lair_dungeon,
            });
        }

        let mut regions: Vec<RegionInfo> = Vec::new();
        let mut settlement_uids: Vec<String> = Vec::new();
        let mut dungeon_uids: Vec<String> = Vec::new();

        // ── Regions → hexes (terrain + encounter + feature links) ───────
        let region_uids: Vec<String> = tx
            .load(&realm_uid)?
            .value["regions"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();

        for region_uid in &region_uids {
            let Ok(region_raw) = tx.load(region_uid) else { continue };
            let region_class = region_raw.value["class"].as_str().unwrap_or("").to_string();
            let terrain = region_class.trim_end_matches("Region").to_string();
            let hex_uids: Vec<String> = region_raw.value["Hexmap"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            let region_name = render_uid!(region_uid)
                .map(|r| {
                    let n = field_str(&r, "Name");
                    if n.is_empty() { field_str(&r, "Title") } else { n }
                })
                .unwrap_or_default();

            let mut hexes: Vec<HexInfo> = Vec::new();
            for hex_uid in &hex_uids {
                let Ok(hex_raw) = tx.load(hex_uid) else { continue };
                let hex_terrain = hex_raw.value["class"]
                    .as_str()
                    .unwrap_or("")
                    .trim_end_matches("Hex")
                    .to_string();
                let first_uid = |arr: &Value| -> Option<String> {
                    arr.as_array()
                        .and_then(|a| a.first())
                        .and_then(|v| v.as_str().map(str::to_string))
                };
                let settlement = first_uid(&hex_raw.value["Settlement"]);
                let dungeon = first_uid(&hex_raw.value["Dungeon"]);
                if let Some(s) = &settlement { settlement_uids.push(s.clone()); }
                if let Some(d) = &dungeon { dungeon_uids.push(d.clone()); }
                // Encounter: render the hex and read its inline Monster.
                let encounter = render_uid!(hex_uid)
                    .and_then(|h| monster_brief(&h["Monster"]));
                // Wilderness feature (landmark): render it and read Name + Description.
                let feature = first_uid(&hex_raw.value["Feature"])
                    .and_then(|fu| render_uid!(&fu))
                    .and_then(|f| {
                        let name = {
                            let n = field_str(&f, "Name");
                            if n.is_empty() { field_str(&f, "Title") } else { n }
                        };
                        if name.is_empty() { return None; }
                        let description = clean_prose(&html_to_text(&field_str(&f, "Description")));
                        Some(FeatureBrief { name: clean_prose(&name), description })
                    });
                hexes.push(HexInfo {
                    uid: hex_uid.clone(),
                    terrain: hex_terrain,
                    encounter,
                    settlement,
                    dungeon,
                    feature,
                });
            }
            regions.push(RegionInfo { name: region_name, terrain, hexes });
        }

        // ── Settlements (summary) ───────────────────────────────────────
        settlement_uids.sort();
        settlement_uids.dedup();
        let mut settlements: Vec<SettlementInfo> = Vec::new();
        for uid in &settlement_uids {
            let Ok(raw) = tx.load(uid) else { continue };
            let kind = raw.value["class"].as_str().unwrap_or("").to_string();
            let r = render_uid!(uid).unwrap_or(Value::Null);
            let name = {
                let n = field_str(&r, "Name");
                if n.is_empty() { field_str(&r, "Title") } else { n }
            };

            // NPCs notáveis: entidades com classe "<Classe>Level<N>" no subtree.
            let mut npcs: Vec<NpcBrief> = Vec::new();
            let mut cand: Vec<String> = Vec::new();
            collect_uids_recursive(&raw.value, &mut cand, 5);
            cand.sort();
            cand.dedup();
            for c in &cand {
                if npcs.len() >= 6 {
                    break;
                }
                let Ok(craw) = tx.load(c) else { continue };
                let cls = craw.value["class"].as_str().unwrap_or("");
                let Some((base, level)) = npc_class_level(cls) else { continue };
                let nr = render_uid!(c).unwrap_or(Value::Null);
                let nname = {
                    let n = field_str(&nr, "Name");
                    if n.is_empty() { field_str(&nr, "Title") } else { n }
                };
                npcs.push(NpcBrief {
                    name: if nname.is_empty() { base.clone() } else { nname },
                    class: base,
                    level,
                    hp: field_str(&nr, "HP"),
                    armour_class: field_str(&nr, "ArmourClass"),
                    thac0: field_str(&nr, "THAC0"),
                    alignment: field_str(&nr, "Alignment"),
                });
            }

            // Caminhada transitiva única pela subárvore do assentamento:
            // colhe ganchos de aventura (quests) + interior (taverna, lojas).
            // O texto/nome só resolve via render_entity. A subárvore "vaza"
            // para assentamentos vizinhos via back-links — por isso taverna/
            // lojas são filtradas por SettlementUUID == este assentamento.
            let mut quests: Vec<QuestBrief> = Vec::new();
            let mut shops: Vec<ShopBrief> = Vec::new();
            let mut tavern: Option<TavernBrief> = None;
            let mut tavern_dish = String::new();
            {
                let mut seenq = std::collections::HashSet::new();
                let mut frontier = vec![uid.clone()];
                while let Some(u) = frontier.pop() {
                    if !seenq.insert(u.clone()) || seenq.len() > 4000 { continue; }
                    let Ok(e) = tx.load(&u) else { continue };
                    let cl = e.value["class"].as_str().unwrap_or("").to_string();

                    if let Some(qkind) = quest_kind(&cl) {
                        if quests.len() < 6 {
                            if let Ok(rq) = render_entity(&render_instance, &mut bp, tx, &e.value, true) {
                                let text = clean_quest_text(&html_to_text(rq["Description"].as_str().unwrap_or("")));
                                if quest_text_ok(&text)
                                    && !quests.iter().any(|q: &QuestBrief| q.text == text)
                                {
                                    quests.push(QuestBrief { kind: qkind, text });
                                }
                            }
                        }
                    } else if cl == "DistrictTavern" && tavern.is_none() {
                        if let Ok(re) = render_entity(&render_instance, &mut bp, tx, &e.value, true) {
                            if re["SettlementUUID"].as_str() == Some(uid.as_str()) {
                                let name = strip_quotes(&field_str(&re, "Title"));
                                let tkind = field_str(&re, "LinkedName");
                                let tkind = tkind.split(" (").next().unwrap_or("Tavern").trim().to_string();
                                if !name.is_empty() {
                                    tavern = Some(TavernBrief { name, kind: tkind, dish: String::new() });
                                }
                            }
                        }
                    } else if cl == "TavernDish" && tavern_dish.is_empty() {
                        if let Ok(re) = render_entity(&render_instance, &mut bp, tx, &e.value, true) {
                            tavern_dish = html_to_text(re["Description"].as_str().unwrap_or(""))
                                .split_whitespace().collect::<Vec<_>>().join(" ");
                        }
                    } else if shops.len() < 8 {
                        // Loja tipada: tem BaseName + Title + pertence a um distrito.
                        if let Ok(re) = render_entity(&render_instance, &mut bp, tx, &e.value, true) {
                            let base = field_str(&re, "BaseName");
                            let title = field_str(&re, "Title");
                            let in_district = re.get("DistrictUUID").and_then(|v| v.as_str()).is_some();
                            let mine = re["SettlementUUID"].as_str() == Some(uid.as_str());
                            if mine && in_district && !base.is_empty() && !title.is_empty()
                                && !shops.iter().any(|s: &ShopBrief| s.name == base)
                            {
                                // CostFactor vive no Distrito; lê via DistrictUUID.
                                let cost_factor = re.get("DistrictUUID")
                                    .and_then(|v| v.as_str())
                                    .and_then(|du| render_uid!(du))
                                    .map(|dr| field_str(&dr, "CostFactor"))
                                    .and_then(|s| s.parse::<f64>().ok())
                                    .filter(|c| *c > 0.0)
                                    .unwrap_or(1.0);
                                shops.push(ShopBrief { name: base, kind: title, cost_factor });
                            }
                        }
                    }

                    let mut ch = Vec::new();
                    collect_uids_recursive(&e.value, &mut ch, 3);
                    for c in ch { if !seenq.contains(&c) { frontier.push(c); } }
                }
                if let Some(t) = tavern.as_mut() {
                    if t.dish.is_empty() && tavern_dish.split_whitespace().count() >= 4 {
                        t.dish = tavern_dish;
                    }
                }
            }

            settlements.push(SettlementInfo {
                uid: uid.clone(),
                name,
                kind,
                region: field_str(&r, "Region"),
                hex: field_str(&r, "HexLink"),
                population: field_str(&r, "Population"),
                npcs,
                quests,
                tavern,
                shops,
            });
        }

        // ── Dungeons + interiors ────────────────────────────────────────
        dungeon_uids.sort();
        dungeon_uids.dedup();
        let mut dungeons: Vec<DungeonInfo> = Vec::new();
        for uid in &dungeon_uids {
            let Ok(raw) = tx.load(uid) else { continue };
            let kind = raw.value["class"].as_str().unwrap_or("").to_string();
            // `map` is stored as a single-element array of the DungeonMap uid.
            let map_uid = match &raw.value["map"] {
                Value::Array(a) => a.first().and_then(|v| v.as_str().map(str::to_string)),
                Value::String(s) => Some(s.clone()),
                Value::Object(o) => o
                    .get("uuid")
                    .or_else(|| o.get("uid"))
                    .and_then(|v| v.as_str().map(str::to_string)),
                _ => None,
            };
            let r = render_uid!(uid).unwrap_or(Value::Null);
            let name = {
                let n = field_str(&r, "Name");
                if n.is_empty() { field_str(&r, "Title") } else { n }
            };

            // Areas: collect DungeonArea entities reachable from the map.
            let mut areas: Vec<AreaInfo> = Vec::new();
            if let Some(map_uid) = &map_uid {
                let mut candidates: Vec<String> = Vec::new();
                if let Ok(map_raw) = tx.load(map_uid) {
                    collect_uids_recursive(&map_raw.value, &mut candidates, 5);
                }
                candidates.sort();
                candidates.dedup();
                for c in &candidates {
                    let Ok(c_raw) = tx.load(c) else { continue };
                    // Rooms are dynamically-named subclasses: "DungeonArea_<uid>".
                    let is_area = c_raw.value["class"]
                        .as_str()
                        .is_some_and(|cl| cl.starts_with("DungeonArea"));
                    if !is_area {
                        continue;
                    }
                    // Grafo real de corredores: as passagens da sala carregam o
                    // número da sala-alvo (Room). Coletamos esses alvos.
                    let mut connections: Vec<i64> = Vec::new();
                    // Salas alcançadas via porta SECRETA (secret_door_*) — exportadas à
                    // parte para o cliente poder gatear/revelar via busca.
                    let mut secret: Vec<i64> = Vec::new();
                    if let Some(o) = c_raw.value.as_object() {
                        for (k, v) in o {
                            let is_secret = k.starts_with("secret_door_");
                            if !k.starts_with("passage_") && !is_secret {
                                continue;
                            }
                            let cu = v.as_array().and_then(|a| a.first()).and_then(|x| x.as_str())
                                .or_else(|| v.as_str());
                            if let Some(cu) = cu {
                                if let Ok(cr) = tx.load(cu) {
                                    if let Some(room) = cr.value["Room"].as_i64() {
                                        if !connections.contains(&room) {
                                            connections.push(room);
                                        }
                                        if is_secret && !secret.contains(&room) {
                                            secret.push(room);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let ar = render_uid!(c).unwrap_or(Value::Null);
                    let title = {
                        let t = field_str(&ar, "RoomType");
                        if t.is_empty() { field_str(&ar, "Title") } else { t }
                    };
                    let description = {
                        let d = field_str(&ar, "Description");
                        if d.is_empty() { field_str(&ar, "Appearance") } else { d }
                    };
                    // Room encounter is nested under the room's Feature; search.
                    let encounter = find_monster(&ar);

                    // Treasure: the room's Feature carries treasure (DungeonRemains /
                    // DungeonTreasureTierN). Hexroll's exact gp is an unresolved
                    // template headless, so we plant a deterministic, tier-scaled
                    // amount (seed-stable via the area uid). Magic-item names, when
                    // they render cleanly, are surfaced separately.
                    let feat_class = ar["Feature"]["class"].as_str().unwrap_or("");
                    let has_treasure = feat_class.starts_with("DungeonTreasure")
                        || feat_class == "DungeonRemains";
                    let tier = field_str(&ar, "FeatureLevelClass")
                        .chars().filter(|ch| ch.is_ascii_digit()).last()
                        .and_then(|ch| ch.to_digit(10)).unwrap_or(1) as i64;
                    let treasure_gold = if has_treasure {
                        let span = (tier as u64) * 40 + 30;
                        tier * 30 + (fnv1a(c) % span) as i64
                    } else {
                        0
                    };
                    let mut treasure_items: Vec<String> = Vec::new();
                    find_magic_items(&ar, &mut treasure_items);

                    // Armadilha da sala (Feature.AreaTrap).
                    let trap = ar["Feature"]["AreaTrap"].as_object().and_then(|t| {
                        let cls = t.get("class").and_then(|c| c.as_str()).unwrap_or("");
                        let desc = html_to_text(
                            t.get("Description").and_then(|d| d.as_str()).unwrap_or(""),
                        );
                        // Pula placeholders "NoTrap" e armadilhas sem descrição.
                        if desc.is_empty() || cls.contains("NoTrap") || cls.contains("None") {
                            None
                        } else {
                            Some(TrapBrief { name: humanize_class(cls), description: desc })
                        }
                    });

                    // Room connectivity is reconstructed client-side via an MST
                    // over room coordinates (hexroll's exact corridor graph lives
                    // in separate entities we don't surface here).
                    areas.push(AreaInfo {
                        number: c_raw.value["RoomNumber"].as_i64().unwrap_or(0),
                        x: c_raw.value["x_coords"].as_i64().unwrap_or(0),
                        y: c_raw.value["y_coords"].as_i64().unwrap_or(0),
                        title,
                        description,
                        encounter,
                        treasure_gold,
                        treasure_items,
                        trap,
                        connections,
                        secret,
                    });
                }
                areas.sort_by_key(|a| a.number);
            }

            // Wandering monster table: DungeonWanderingMonsters.monsters (10 rolls),
            // deduped by name. Surfaced so the client rolls the dungeon's own table.
            let wandering = {
                let wu = match &raw.value["WanderingMonsters"] {
                    Value::Array(a) => a.first().and_then(|v| v.as_str().map(str::to_string)),
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                };
                let mut out: Vec<MonsterBrief> = Vec::new();
                if let Some(wu) = wu {
                    if let Some(w) = render_uid!(&wu) {
                        if let Some(arr) = w["monsters"].as_array() {
                            let mut seen = std::collections::HashSet::new();
                            for m in arr {
                                if let Some(mb) = monster_brief(m).or_else(|| find_monster(m)) {
                                    if seen.insert(mb.name.clone()) { out.push(mb); }
                                }
                            }
                        }
                    }
                }
                out
            };

            dungeons.push(DungeonInfo {
                uid: uid.clone(),
                name,
                kind,
                region: field_str(&r, "Region"),
                hex: field_str(&r, "HexLink"),
                entrances: field_str(&r, "Entrances"),
                areas,
                wandering,
            });
        }

        Ok(GenerateResponse {
            version: 2,
            seed,
            sandbox_id: sid.clone(),
            realm,
            regions,
            settlements,
            dungeons,
            factions,
        })
    })
}
