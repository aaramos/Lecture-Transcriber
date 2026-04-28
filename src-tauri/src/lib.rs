use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::{env, fs};
use tauri::{Emitter, Manager};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

const EVENT_PREFIX: &str = "__LECTURE_PROCESSOR_EVENT__ ";
const KEYCHAIN_SERVICE: &str = "Lecture Processor";
const SLIDE_TEMP_PREFIX: &str = "lecture-slides-";
const OUTPUT_LOCK_FILE: &str = ".lecture_processor.lock";
const CONTROL_FILE: &str = ".lecture_processor_control.json";
const NORMALIZED_WORK_FILE: &str = ".normalized_work.mp4";
const TRANSCRIPTION_AUDIO_FILE: &str = ".transcription_audio.wav";
const FFMPEG_TEMP_SUFFIX: &str = ".ffmpeg.tmp";
const ATOMIC_TEMP_FILES: [&str; 9] = [
    ".batch.json.tmp",
    ".batch_error.txt.tmp",
    ".batch_summary.txt.tmp",
    ".index.html.tmp",
    ".lecture.json.tmp",
    "..lecture_processor_control.json.tmp",
    ".processing_log.txt.tmp",
    ".transcript.srt.tmp",
    ".transcript.txt.tmp",
];

#[derive(Clone)]
struct ActiveProcess {
    pid: u32,
    cancelled: bool,
}

#[derive(Default)]
struct AppState {
    active_process: Arc<Mutex<Option<ActiveProcess>>>,
    metrics_state: Arc<Mutex<MetricsState>>,
}

#[derive(Default)]
struct MetricsState {
    last_gpu_time_ns: Option<u64>,
    last_gpu_sample: Option<Instant>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FolderScan {
    folder_mode: String,
    mov_count: usize,
    mov_files: Vec<String>,
    processed_count: usize,
    lecture_files: Vec<String>,
    already_processed_files: Vec<String>,
    already_enhanced_files: Vec<String>,
    output_dir: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessRequest {
    #[serde(default = "default_process_mode")]
    mode: String,
    input_dir: String,
    output_dir: String,
    recording_speed: String,
    confirm_normalization: bool,
    concurrent_files: u8,
    save_normalized_video: bool,
    audio_quality: String,
    transcription_engine: String,
    transcription_quality: String,
    whisper_model: String,
    slide_sensitivity: String,
    ai_provider: String,
    ai_model: String,
    min_duration: f64,
    skipped_files: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProcessResponse {
    exit_code: i32,
    stdout: String,
    stderr: String,
    output_dir: String,
    cancelled: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CancelResponse {
    cancelled: bool,
    message: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CleanupTempResponse {
    deleted: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiKeyStatus {
    saved: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SystemMetrics {
    cpu_percent: Option<f64>,
    gpu_percent: Option<f64>,
    memory_used_gb: Option<f64>,
    cpu_status: String,
    gpu_status: String,
    memory_status: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DependencyStatus {
    ready: bool,
    message: String,
    detail: String,
    runtime_dir: String,
    processor_cli: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DependencyEvent {
    kind: String,
    title: String,
    detail: String,
    progress: u8,
    step: u8,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileControlRequest {
    output_dir: String,
    source: String,
    action: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileControlResponse {
    status: String,
}

#[derive(Clone)]
struct ProcessedLectureScan {
    source_name: String,
    enhanced: bool,
}

fn default_process_mode() -> String {
    "process".to_string()
}

#[tauri::command]
fn choose_folder() -> Result<Option<String>, String> {
    Ok(rfd::FileDialog::new()
        .pick_folder()
        .map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
fn default_output_dir(input_dir: String) -> Result<String, String> {
    let input = PathBuf::from(input_dir);
    Ok(default_output_dir_for_path(&input)
        .to_string_lossy()
        .to_string())
}

fn default_output_dir_for_path(input: &Path) -> PathBuf {
    let name = input
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{name}_processed"))
}

#[tauri::command]
fn scan_folder(input_dir: String, output_dir: Option<String>) -> Result<FolderScan, String> {
    let input_path = PathBuf::from(&input_dir);
    let processed_lectures = scan_processed_lectures(&input_path)?;
    if !processed_lectures.is_empty() {
        let mut lecture_files = processed_lectures
            .iter()
            .map(|lecture| lecture.source_name.clone())
            .collect::<Vec<_>>();
        lecture_files.sort_by_key(|value| value.to_lowercase());
        let mut already_enhanced_files = processed_lectures
            .iter()
            .filter(|lecture| lecture.enhanced)
            .map(|lecture| lecture.source_name.clone())
            .collect::<Vec<_>>();
        already_enhanced_files.sort_by_key(|value| value.to_lowercase());
        return Ok(FolderScan {
            folder_mode: "processed".to_string(),
            mov_count: 0,
            mov_files: Vec::new(),
            processed_count: lecture_files.len(),
            lecture_files,
            already_processed_files: Vec::new(),
            already_enhanced_files,
            output_dir: input_path.to_string_lossy().to_string(),
        });
    }

    let entries =
        std::fs::read_dir(&input_dir).map_err(|error| format!("Could not read folder: {error}"))?;
    let mut mov_files = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|value| value.to_str())
                .map(|extension| extension.eq_ignore_ascii_case("mov"))
                .unwrap_or(false)
        })
        .filter_map(|entry| entry.file_name().to_str().map(|value| value.to_string()))
        .collect::<Vec<_>>();
    mov_files.sort_by_key(|value| value.to_lowercase());
    let output_path = output_dir
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| default_output_dir_for_path(&input_path));
    let already_processed_files =
        already_processed_files_for_source(&input_path, &output_path, &mov_files);
    Ok(FolderScan {
        folder_mode: "source".to_string(),
        mov_count: mov_files.len(),
        mov_files,
        processed_count: 0,
        lecture_files: Vec::new(),
        already_processed_files,
        already_enhanced_files: Vec::new(),
        output_dir: output_path.to_string_lossy().to_string(),
    })
}

fn scan_processed_lectures(folder: &Path) -> Result<Vec<ProcessedLectureScan>, String> {
    if !folder.is_dir() {
        return Ok(Vec::new());
    }
    let mut lecture_jsons = Vec::new();
    let root_artifact = folder.join("lecture.json");
    if root_artifact.exists() {
        lecture_jsons.push(root_artifact);
    } else {
        let entries =
            std::fs::read_dir(folder).map_err(|error| format!("Could not read folder: {error}"))?;
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path().join("lecture.json");
            if path.exists() {
                lecture_jsons.push(path);
            }
        }
    }
    lecture_jsons.sort_by_key(|path| {
        path.parent()
            .and_then(|value| value.file_name())
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_lowercase()
    });

    let mut lectures = Vec::new();
    for path in lecture_jsons {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if !processed_lecture_can_be_enriched(&payload) {
            continue;
        }
        let source_name = payload
            .get("source")
            .and_then(|value| value.get("filename"))
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .or_else(|| {
                path.parent()
                    .and_then(|value| value.file_name())
                    .and_then(|value| value.to_str())
                    .map(|value| format!("{value}.mov"))
            })
            .unwrap_or_else(|| "lecture.mov".to_string());
        let enhanced = !payload
            .get("enrichment")
            .map(|value| value.is_null())
            .unwrap_or(true);
        lectures.push(ProcessedLectureScan {
            source_name,
            enhanced,
        });
    }
    Ok(lectures)
}

fn processed_lecture_can_be_enriched(payload: &serde_json::Value) -> bool {
    let processing = payload
        .get("processing")
        .unwrap_or(&serde_json::Value::Null);
    if processing.get("status").and_then(|value| value.as_str()) == Some("completed") {
        return true;
    }
    if processing.get("status").and_then(|value| value.as_str()) != Some("failed") {
        return false;
    }
    if processing
        .get("failure_step")
        .and_then(|value| value.as_str())
        != Some("Enrich")
    {
        return false;
    }
    let transcript = payload
        .get("transcript")
        .unwrap_or(&serde_json::Value::Null);
    transcript
        .get("text")
        .and_then(|value| value.as_str())
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
        || transcript
            .get("segments")
            .and_then(|value| value.as_array())
            .map(|value| !value.is_empty())
            .unwrap_or(false)
        || payload
            .get("slides")
            .and_then(|value| value.as_array())
            .map(|value| !value.is_empty())
            .unwrap_or(false)
}

fn already_processed_files_for_source(
    input_dir: &Path,
    output_dir: &Path,
    mov_files: &[String],
) -> Vec<String> {
    let mut seen = std::collections::BTreeMap::<String, usize>::new();
    let mut processed = Vec::new();
    for filename in mov_files {
        let source = input_dir.join(filename);
        let stem = source
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or(filename);
        let base = safe_folder_name(stem);
        let key = base.to_lowercase();
        let count = seen.entry(key).or_insert(0);
        *count += 1;
        let folder_name = if *count == 1 {
            base
        } else {
            format!("{}_{}", base, *count)
        };
        let lecture_json = output_dir.join(folder_name).join("lecture.json");
        if completed_artifact_matches_source(&lecture_json, filename) {
            processed.push(filename.clone());
        }
    }
    processed
}

fn completed_artifact_matches_source(lecture_json: &Path, source_filename: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(lecture_json) else {
        return false;
    };
    let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    if payload
        .get("processing")
        .and_then(|value| value.get("status"))
        .and_then(|value| value.as_str())
        != Some("completed")
    {
        return false;
    }
    payload
        .get("source")
        .and_then(|value| value.get("filename"))
        .and_then(|value| value.as_str())
        .map(|value| value == source_filename)
        .unwrap_or(true)
}

fn safe_folder_name(stem: &str) -> String {
    let mut value = String::new();
    let mut last_was_separator = false;
    for character in stem.chars() {
        if character.is_ascii_alphanumeric()
            || character == '.'
            || character == '_'
            || character == '-'
        {
            value.push(character);
            last_was_separator = false;
        } else if !last_was_separator {
            value.push('_');
            last_was_separator = true;
        }
    }
    let trimmed = value.trim_matches(&['.', '_', '-'][..]).to_string();
    if trimmed.is_empty() {
        "lecture".to_string()
    } else {
        trimmed
    }
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let mut command = platform_open_command(&path);
    let status = command
        .status()
        .map_err(|error| format!("Could not open output folder: {error}"))?;
    if !status.success() {
        return Err("Could not open output folder.".to_string());
    }
    Ok(())
}

#[tauri::command]
fn cleanup_temp_files(output_dir: Option<String>) -> Result<CleanupTempResponse, String> {
    Ok(CleanupTempResponse {
        deleted: cleanup_temp_files_impl(output_dir.as_deref().map(Path::new)),
    })
}

#[tauri::command]
fn save_api_key(provider: String, api_key: String) -> Result<(), String> {
    let account = keychain_account(&provider)?;
    let value = api_key.trim();
    if value.is_empty() {
        return Err("API key cannot be empty.".to_string());
    }
    let status = Command::new("security")
        .args([
            "add-generic-password",
            "-a",
            &account,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
            value,
            "-U",
        ])
        .status()
        .map_err(|error| format!("Could not save API key to Keychain: {error}"))?;
    if !status.success() {
        return Err("Could not save API key to Keychain.".to_string());
    }
    Ok(())
}

#[tauri::command]
fn has_api_key(provider: String) -> Result<ApiKeyStatus, String> {
    Ok(ApiKeyStatus {
        saved: read_api_key(&provider)?.is_some(),
    })
}

#[tauri::command]
fn system_metrics(state: tauri::State<'_, AppState>) -> Result<SystemMetrics, String> {
    let cpu_percent = sample_cpu_percent();
    let memory_used_gb = sample_memory_used_gb();
    let gpu_percent = sample_gpu_percent(&state.metrics_state);

    Ok(SystemMetrics {
        cpu_status: metric_status(cpu_percent, "Unavailable"),
        gpu_status: metric_status(gpu_percent, "Warming up"),
        memory_status: metric_status(memory_used_gb, "Unavailable"),
        cpu_percent,
        gpu_percent,
        memory_used_gb,
    })
}

#[tauri::command]
fn dependency_status(app: tauri::AppHandle) -> Result<DependencyStatus, String> {
    dependency_status_impl(&app)
}

#[tauri::command]
async fn setup_dependencies(app: tauri::AppHandle) -> Result<DependencyStatus, String> {
    let app_for_setup = app.clone();
    tauri::async_runtime::spawn_blocking(move || setup_dependencies_impl(app_for_setup))
        .await
        .map_err(|error| format!("Dependency setup task failed: {error}"))?
}

#[tauri::command]
async fn process_batch(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    request: ProcessRequest,
) -> Result<ProcessResponse, String> {
    let active_process = Arc::clone(&state.active_process);
    tauri::async_runtime::spawn_blocking(move || run_process_batch(app, active_process, request))
        .await
        .map_err(|error| format!("Processor task failed: {error}"))?
}

#[tauri::command]
fn cancel_batch(state: tauri::State<'_, AppState>) -> Result<CancelResponse, String> {
    let process_to_cancel = mark_active_process_cancelled(&state.active_process);

    let Some((pid, previous_cancelled)) = process_to_cancel else {
        return Ok(CancelResponse {
            cancelled: false,
            message: "No batch is currently running.".to_string(),
        });
    };

    if let Err(error) = terminate_process_tree(pid) {
        let mut active = state
            .active_process
            .lock()
            .unwrap_or_else(|lock_error| lock_error.into_inner());
        if let Some(process) = active.as_mut() {
            if process.pid == pid {
                process.cancelled = previous_cancelled;
            }
        }
        return Err(error);
    }

    Ok(CancelResponse {
        cancelled: true,
        message: "Cancellation requested.".to_string(),
    })
}

#[tauri::command]
fn update_file_control(request: FileControlRequest) -> Result<FileControlResponse, String> {
    let output_dir = PathBuf::from(&request.output_dir);
    if output_dir.as_os_str().is_empty() {
        return Err("Output folder is not set.".to_string());
    }
    if request.source.trim().is_empty() {
        return Err("Video name is missing.".to_string());
    }

    let control_file = output_dir.join(CONTROL_FILE);
    match request.action.as_str() {
        "skip" => update_control_file(&control_file, &[request.source.as_str()], &[])?,
        "stop" => update_control_file(&control_file, &[], &[request.source.as_str()])?,
        _ => return Err("Unsupported file control action.".to_string()),
    }
    Ok(FileControlResponse {
        status: request.action,
    })
}

fn metric_status(value: Option<f64>, fallback: &str) -> String {
    if value.is_some() {
        "Live".to_string()
    } else {
        fallback.to_string()
    }
}

fn sample_cpu_percent() -> Option<f64> {
    let output = Command::new("ps")
        .args(["-A", "-o", "%cpu="])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let total_process_cpu = text
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    let logical_cpus = thread::available_parallelism().ok()?.get() as f64;
    Some(round_one(
        (total_process_cpu / logical_cpus).clamp(0.0, 100.0),
    ))
}

fn sample_memory_used_gb() -> Option<f64> {
    let output = Command::new("vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let page_size = parse_vm_page_size(&text)?;
    let active = parse_vm_pages(&text, "Pages active")?;
    let wired = parse_vm_pages(&text, "Pages wired down")?;
    let compressed = parse_vm_pages(&text, "Pages occupied by compressor").unwrap_or(0);
    let used_bytes = (active + wired + compressed) as f64 * page_size as f64;
    Some(round_one(used_bytes / 1_073_741_824.0))
}

fn sample_gpu_percent(metrics_state: &Arc<Mutex<MetricsState>>) -> Option<f64> {
    let total_gpu_time_ns = sample_gpu_accumulated_time_ns()?;
    let now = Instant::now();
    let mut state = metrics_state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let percent = match (state.last_gpu_time_ns, state.last_gpu_sample) {
        (Some(previous_time), Some(previous_sample)) if total_gpu_time_ns >= previous_time => {
            let elapsed_ns = now.duration_since(previous_sample).as_nanos() as f64;
            if elapsed_ns > 0.0 {
                Some(round_one(
                    ((total_gpu_time_ns - previous_time) as f64 / elapsed_ns * 100.0)
                        .clamp(0.0, 100.0),
                ))
            } else {
                None
            }
        }
        _ => None,
    };
    state.last_gpu_time_ns = Some(total_gpu_time_ns);
    state.last_gpu_sample = Some(now);
    percent
}

fn sample_gpu_accumulated_time_ns() -> Option<u64> {
    let output = Command::new("/usr/sbin/ioreg")
        .args(["-l", "-w", "0", "-c", "AGXDeviceUserClient"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let values = parse_accumulated_gpu_times(&text);
    if values.is_empty() {
        return None;
    }
    Some(values.into_iter().sum())
}

fn parse_vm_page_size(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.contains("page size of"))?;
    let value = line.split("page size of").nth(1)?.split("bytes").next()?;
    value.trim().parse::<u64>().ok()
}

fn parse_vm_pages(text: &str, key: &str) -> Option<u64> {
    let line = text
        .lines()
        .find(|line| line.trim_start().starts_with(key))?;
    let value = line.split(':').nth(1)?;
    value
        .trim()
        .trim_end_matches('.')
        .replace('.', "")
        .parse::<u64>()
        .ok()
}

fn parse_accumulated_gpu_times(text: &str) -> Vec<u64> {
    let marker = "\"accumulatedGPUTime\"=";
    let mut values = Vec::new();
    let mut remaining = text;
    while let Some(index) = remaining.find(marker) {
        let after_marker = &remaining[index + marker.len()..];
        let digits = after_marker
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect::<String>();
        if let Ok(value) = digits.parse::<u64>() {
            values.push(value);
        }
        remaining = after_marker;
    }
    values
}

fn round_one(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::{
        parse_accumulated_gpu_times, parse_vm_page_size, parse_vm_pages,
        processed_lecture_can_be_enriched,
    };
    use serde_json::json;

    #[test]
    fn parses_vm_stat_values() {
        let sample = r#"Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages active:                                 628839.
Pages wired down:                             185384.
Pages occupied by compressor:                  87041.
"#;

        assert_eq!(parse_vm_page_size(sample), Some(16_384));
        assert_eq!(parse_vm_pages(sample, "Pages active"), Some(628_839));
        assert_eq!(parse_vm_pages(sample, "Pages wired down"), Some(185_384));
        assert_eq!(
            parse_vm_pages(sample, "Pages occupied by compressor"),
            Some(87_041)
        );
    }

    #[test]
    fn parses_gpu_accumulated_times() {
        let sample = r#"
          "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=1200},{"API"="Metal","accumulatedGPUTime"=3400})
          "AppUsage" = ({"API"="Metal","lastSubmittedTime"=0,"accumulatedGPUTime"=0})
        "#;

        assert_eq!(parse_accumulated_gpu_times(sample), vec![1200, 3400, 0]);
    }

    #[test]
    fn enrichable_processed_lecture_includes_enrich_failures() {
        let payload = json!({
            "processing": {
                "status": "failed",
                "failure_step": "Enrich"
            },
            "transcript": {
                "text": "ready for retry"
            },
            "slides": []
        });

        assert!(processed_lecture_can_be_enriched(&payload));
    }
}

fn mark_active_process_cancelled(
    active_process: &Arc<Mutex<Option<ActiveProcess>>>,
) -> Option<(u32, bool)> {
    let mut active = active_process
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Some(process) = active.as_mut() {
        let previous_cancelled = process.cancelled;
        process.cancelled = true;
        Some((process.pid, previous_cancelled))
    } else {
        None
    }
}

fn terminate_active_process_for_shutdown(active_process: &Arc<Mutex<Option<ActiveProcess>>>) {
    if let Some((pid, _previous_cancelled)) = mark_active_process_cancelled(active_process) {
        let _ = terminate_process_tree(pid);
    }
}

fn run_process_batch(
    app: tauri::AppHandle,
    active_process: Arc<Mutex<Option<ActiveProcess>>>,
    request: ProcessRequest,
) -> Result<ProcessResponse, String> {
    let enhance_mode = request.mode == "enhance";
    if !enhance_mode && request.recording_speed == "2x" && !request.confirm_normalization {
        return Err("2x normalization requires confirmation.".to_string());
    }
    if request.concurrent_files == 0 || request.concurrent_files > 3 {
        return Err("Concurrent files must be between 1 and 3.".to_string());
    }
    if !enhance_mode && request.output_dir.trim().is_empty() {
        return Err("Choose an output folder before starting.".to_string());
    }
    let ai_api_key = if request.ai_provider == "gemini" {
        Some(
            read_api_key("gemini")?
                .ok_or_else(|| "Gemini needs an API key saved in Settings.".to_string())?,
        )
    } else {
        None
    };
    if active_process
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .is_some()
    {
        return Err("A batch is already running.".to_string());
    }

    let project_root = find_project_root(Some(&app))?;
    let cli = processor_cli_path(&app, &project_root)?;
    if !cli.exists() {
        return Err(format!(
            "Processor CLI was not found at {}. Run the first-run dependency installer.",
            cli.display()
        ));
    }

    let mut args = if enhance_mode {
        vec![
            "enrich-batch".to_string(),
            request.input_dir.clone(),
            "--ai-provider".to_string(),
            request.ai_provider.clone(),
            "--ai-model".to_string(),
            request.ai_model.clone(),
            "--concurrent".to_string(),
            request.concurrent_files.to_string(),
        ]
    } else {
        let control_file = Path::new(&request.output_dir).join(CONTROL_FILE);
        let _ = std::fs::remove_file(&control_file);
        if !request.skipped_files.is_empty() {
            let skip_refs = request
                .skipped_files
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            update_control_file(&control_file, &skip_refs, &[])?;
        }

        vec![
            "process".to_string(),
            request.input_dir.clone(),
            "--output".to_string(),
            request.output_dir.clone(),
            "--recording-speed".to_string(),
            request.recording_speed.clone(),
            "--concurrent".to_string(),
            request.concurrent_files.to_string(),
            "--audio-quality".to_string(),
            request.audio_quality.clone(),
            "--transcription-engine".to_string(),
            request.transcription_engine.clone(),
            "--transcription-quality".to_string(),
            request.transcription_quality.clone(),
            "--whisper-model".to_string(),
            request.whisper_model.clone(),
            "--slide-sensitivity".to_string(),
            request.slide_sensitivity.clone(),
            "--ai-provider".to_string(),
            request.ai_provider.clone(),
            "--ai-model".to_string(),
            request.ai_model.clone(),
            "--min-duration".to_string(),
            request.min_duration.to_string(),
            "--control-file".to_string(),
            control_file.to_string_lossy().to_string(),
        ]
    };

    for file in request
        .skipped_files
        .iter()
        .filter(|value| !value.trim().is_empty())
    {
        args.push("--skip-file".to_string());
        args.push(file.clone());
    }
    args.push("--json-events".to_string());

    if !enhance_mode && is_apple_silicon() {
        args.push("--apple-silicon".to_string());
        args.push("--slide-backend".to_string());
        args.push("ffmpeg".to_string());
    }

    let whisper_cpp_model_dir = env::var("LECTURE_PROCESSOR_WHISPER_CPP_MODEL_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| default_whisper_cpp_model_dir(&project_root, &request.whisper_model));
    if !enhance_mode {
        if let Some(model_dir) = whisper_cpp_model_dir {
            args.push("--whisper-cpp-model-dir".to_string());
            args.push(model_dir);
        }
    }

    if !enhance_mode
        && env::var("LECTURE_PROCESSOR_REQUIRE_WHISPER_CPP_COREML")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    {
        args.push("--require-whisper-cpp-coreml".to_string());
    }

    if !enhance_mode && request.confirm_normalization {
        args.push("--confirm-normalization".to_string());
    }

    if !enhance_mode && !request.save_normalized_video {
        args.push("--no-save-normalized-video".to_string());
    }

    let path = tool_path(&app, &project_root);
    let mut command = Command::new(cli);
    command
        .args(args)
        .current_dir(&project_root)
        .env("PATH", path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if env::var_os("PYWHISPERCPP_FLASH_ATTN").is_none() {
        command.env("PYWHISPERCPP_FLASH_ATTN", "0");
    }
    if env::var_os("PYWHISPERCPP_USE_GPU").is_none() {
        command.env("PYWHISPERCPP_USE_GPU", "0");
    }
    if let Some(api_key) = ai_api_key {
        command.env("GEMINI_API_KEY", api_key);
    }

    #[cfg(unix)]
    command.process_group(0);

    let mut child = command
        .spawn()
        .map_err(|error| format!("Could not start processor: {error}"))?;
    let child_pid = child.id();

    {
        let mut active = active_process
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *active = Some(ActiveProcess {
            pid: child_pid,
            cancelled: false,
        });
    }

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Could not capture processor output.".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Could not capture processor errors.".to_string())?;

    let stdout_buffer = Arc::new(Mutex::new(String::new()));
    let stderr_buffer = Arc::new(Mutex::new(String::new()));

    let stdout_handle = {
        let buffer = Arc::clone(&stdout_buffer);
        let app_handle = app.clone();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(raw_event) = line.strip_prefix(EVENT_PREFIX) {
                    match serde_json::from_str::<serde_json::Value>(raw_event) {
                        Ok(payload) => {
                            let _ = app_handle.emit("processor-event", payload);
                        }
                        Err(_) => append_line(&buffer, &line),
                    }
                } else {
                    append_line(&buffer, &line);
                }
            }
        })
    };

    let stderr_handle = {
        let buffer = Arc::clone(&stderr_buffer);
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                append_line(&buffer, &line);
            }
        })
    };

    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            clear_active_process(&active_process, child_pid);
            return Err(format!("Processor failed while running: {error}"));
        }
    };
    let cancelled = clear_active_process(&active_process, child_pid);
    let _ = stdout_handle.join();
    let _ = stderr_handle.join();
    let stdout_text = stdout_buffer
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone();
    let stderr_text = stderr_buffer
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone();

    Ok(ProcessResponse {
        exit_code: if cancelled {
            130
        } else {
            status.code().unwrap_or(1)
        },
        stdout: stdout_text,
        stderr: stderr_text,
        output_dir: if enhance_mode {
            request.input_dir
        } else {
            request.output_dir
        },
        cancelled,
    })
}

fn clear_active_process(active_process: &Arc<Mutex<Option<ActiveProcess>>>, pid: u32) -> bool {
    let mut active = active_process
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if active.as_ref().map(|process| process.pid) != Some(pid) {
        return false;
    }
    active
        .take()
        .map(|process| process.cancelled)
        .unwrap_or(false)
}

#[cfg(unix)]
fn terminate_process_tree(pid: u32) -> Result<(), String> {
    let process_group = format!("-{pid}");
    let status = Command::new("kill")
        .args(["-TERM", &process_group])
        .status()
        .map_err(|error| format!("Could not request cancellation: {error}"))?;
    if !status.success() && !process_group_is_running(pid) {
        return Ok(());
    }
    if !status.success() {
        return Err("Could not request cancellation for the running batch.".to_string());
    }

    for _ in 0..20 {
        if !process_group_is_running(pid) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }

    let kill_status = Command::new("kill")
        .args(["-KILL", &process_group])
        .status()
        .map_err(|error| format!("Could not force-stop the running batch: {error}"))?;
    if kill_status.success() || !process_group_is_running(pid) {
        Ok(())
    } else {
        Err("Could not force-stop the running batch.".to_string())
    }
}

#[cfg(windows)]
fn terminate_process_tree(pid: u32) -> Result<(), String> {
    let pid = pid.to_string();
    let status = Command::new("taskkill")
        .args(["/PID", &pid, "/T", "/F"])
        .status()
        .map_err(|error| format!("Could not request cancellation: {error}"))?;
    if status.success() {
        return Ok(());
    }
    Err("Could not request cancellation for the running batch.".to_string())
}

#[cfg(not(any(unix, windows)))]
fn terminate_process_tree(_pid: u32) -> Result<(), String> {
    Err("Cancellation is not supported on this platform yet.".to_string())
}

fn append_line(buffer: &Arc<Mutex<String>>, line: &str) {
    let mut value = buffer.lock().unwrap_or_else(|error| error.into_inner());
    value.push_str(line);
    value.push('\n');
}

fn platform_open_command(path: &str) -> Command {
    #[cfg(target_os = "macos")]
    {
        let mut command = Command::new("open");
        command.arg(path);
        command
    }

    #[cfg(target_os = "windows")]
    {
        let mut command = Command::new("cmd");
        command.args(["/C", "start", "", path]);
        command
    }

    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        let mut command = Command::new("xdg-open");
        command.arg(path);
        command
    }
}

fn keychain_account(provider: &str) -> Result<String, String> {
    match provider {
        "gemini" => Ok("gemini_api_key".to_string()),
        _ => Err(format!("Unsupported API key provider: {provider}")),
    }
}

fn read_api_key(provider: &str) -> Result<Option<String>, String> {
    let account = keychain_account(provider)?;
    let output = Command::new("security")
        .args([
            "find-generic-password",
            "-a",
            &account,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
        ])
        .output()
        .map_err(|error| format!("Could not read API key from Keychain: {error}"))?;
    if output.status.success() {
        let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return Ok((!key.is_empty()).then_some(key));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("could not be found") || stderr.contains("-25300") {
        return Ok(None);
    }
    Err("Could not read API key from Keychain.".to_string())
}

fn is_apple_silicon() -> bool {
    #[cfg(target_os = "macos")]
    {
        Command::new("sysctl")
            .args(["-n", "hw.optional.arm64"])
            .output()
            .map(|output| {
                output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == "1"
            })
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

fn dependency_status_impl(app: &tauri::AppHandle) -> Result<DependencyStatus, String> {
    let runtime_dir = runtime_dir(app)?;
    let project_root = find_project_root(Some(app)).ok();
    let processor_cli = project_root
        .as_ref()
        .and_then(|root| {
            let cli = root.join(".venv/bin/lecture-processor");
            cli.exists().then_some(cli)
        })
        .unwrap_or_else(|| runtime_dir.join(".venv/bin/lecture-processor"));
    let ready = processor_cli.exists();
    Ok(DependencyStatus {
        ready,
        message: if ready {
            "Dependencies ready".to_string()
        } else {
            "Dependencies need setup".to_string()
        },
        detail: if ready {
            "The local processor runtime is available.".to_string()
        } else {
            "Install the processor runtime before starting a batch.".to_string()
        },
        runtime_dir: runtime_dir.to_string_lossy().to_string(),
        processor_cli: processor_cli.to_string_lossy().to_string(),
    })
}

fn setup_dependencies_impl(app: tauri::AppHandle) -> Result<DependencyStatus, String> {
    emit_dependency_event(
        &app,
        "running",
        "Preparing setup",
        "Creating the app support folder.",
        5,
        0,
    );
    let runtime_dir = runtime_dir(&app)?;
    fs::create_dir_all(&runtime_dir)
        .map_err(|error| format!("Could not create runtime folder: {error}"))?;

    emit_dependency_event(
        &app,
        "running",
        "Finding processor source",
        "Locating the bundled Python processor.",
        14,
        0,
    );
    let project_root = find_project_root(Some(&app))?;

    let venv_dir = runtime_dir.join(".venv");
    let venv_python = venv_dir.join("bin/python");
    if !venv_python.exists() {
        emit_dependency_event(
            &app,
            "running",
            "Creating Python runtime",
            "Building a private Python environment for Lecture Processor.",
            28,
            1,
        );
        let mut command = Command::new("python3");
        command.args(["-m", "venv"]).arg(&venv_dir);
        run_setup_command(&mut command, "create Python runtime")?;
    }

    emit_dependency_event(
        &app,
        "running",
        "Updating installer tools",
        "Updating pip, setuptools, and wheel.",
        42,
        2,
    );
    let mut upgrade = Command::new(&venv_python);
    upgrade.args([
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    ]);
    run_setup_command(&mut upgrade, "update Python installer tools")?;

    emit_dependency_event(
        &app,
        "running",
        "Installing dependencies",
        "Downloading and installing transcription, slide, and AI packages.",
        58,
        2,
    );
    let install_target = format!(
        "{}[transcription,slides,ai]",
        project_root.to_string_lossy()
    );
    let mut install = Command::new(&venv_python);
    install.args(["-m", "pip", "install"]).arg(install_target);
    run_setup_command(&mut install, "install processor dependencies")?;

    emit_dependency_event(
        &app,
        "running",
        "Verifying processor",
        "Checking that the command line processor starts.",
        86,
        3,
    );
    let cli = processor_cli_path(&app, &project_root)?;
    let mut verify = Command::new(&cli);
    verify
        .arg("--help")
        .env("PATH", tool_path(&app, &project_root));
    run_setup_command(&mut verify, "verify processor command")?;

    emit_dependency_event(
        &app,
        "complete",
        "Setup complete",
        "Dependencies are installed and ready.",
        100,
        4,
    );
    dependency_status_impl(&app)
}

fn run_setup_command(command: &mut Command, action: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("Could not {action}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    Err(format!("Could not {action}: {}", last_lines(detail, 8)))
}

fn last_lines(text: &str, count: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(count);
    lines[start..].join("\n")
}

fn emit_dependency_event(
    app: &tauri::AppHandle,
    kind: &str,
    title: &str,
    detail: &str,
    progress: u8,
    step: u8,
) {
    let _ = app.emit(
        "dependency-event",
        DependencyEvent {
            kind: kind.to_string(),
            title: title.to_string(),
            detail: detail.to_string(),
            progress,
            step,
        },
    );
}

fn runtime_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Could not find app support folder: {error}"))?
        .join("runtime"))
}

fn processor_cli_path(app: &tauri::AppHandle, project_root: &Path) -> Result<PathBuf, String> {
    let project_cli = project_root.join(".venv/bin/lecture-processor");
    if project_cli.exists() {
        return Ok(project_cli);
    }
    Ok(runtime_dir(app)?.join(".venv/bin/lecture-processor"))
}

fn find_project_root(app: Option<&tauri::AppHandle>) -> Result<PathBuf, String> {
    if let Ok(value) = env::var("LECTURE_PROCESSOR_ROOT") {
        let path = PathBuf::from(value);
        if path.join("pyproject.toml").exists() {
            return Ok(path);
        }
    }

    let mut starts = Vec::new();
    if let Ok(current) = env::current_dir() {
        starts.push(current);
    }
    if let Ok(exe) = env::current_exe() {
        starts.push(exe);
    }
    if let Some(app) = app {
        if let Ok(resource_dir) = app.path().resource_dir() {
            starts.push(resource_dir.join("_up_"));
            starts.push(resource_dir);
        }
    }

    for start in starts {
        for candidate in start.ancestors() {
            if candidate.join("pyproject.toml").exists()
                && candidate.join("src/lecture_processor").exists()
            {
                return Ok(candidate.to_path_buf());
            }
        }
    }

    Err("Could not locate the Lecture Processor project root.".to_string())
}

fn default_whisper_cpp_model_dir(project_root: &Path, model_name: &str) -> Option<String> {
    let model_dir = project_root.join(".models/whisper-cpp");
    if !model_dir.is_dir() {
        return None;
    }

    let model_file = if model_name.ends_with(".bin") {
        model_dir.join(model_name)
    } else {
        model_dir.join(format!("ggml-{model_name}.bin"))
    };

    if model_file.exists() {
        Some(model_dir.to_string_lossy().to_string())
    } else {
        None
    }
}

fn tool_path(app: &tauri::AppHandle, project_root: &Path) -> String {
    let mut paths = vec![
        runtime_dir(app)
            .unwrap_or_else(|_| project_root.to_path_buf())
            .join(".venv/bin"),
        project_root.join(".tools/darwin_arm64"),
        project_root.join(".venv/bin"),
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
    ];
    if let Some(existing) = env::var_os("PATH") {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths)
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

fn cleanup_temp_files_impl(output_dir: Option<&Path>) -> usize {
    let mut deleted = cleanup_slide_temp_dirs();
    if let Some(path) = output_dir {
        deleted += cleanup_output_temp_files(path);
    }
    deleted
}

fn update_control_file(path: &Path, skip: &[&str], stop: &[&str]) -> Result<(), String> {
    let mut skip_set = std::collections::BTreeSet::new();
    let mut stop_set = std::collections::BTreeSet::new();

    if path.exists() {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("Could not read file control state: {error}"))?;
        if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(values) = payload.get("skip").and_then(|value| value.as_array()) {
                skip_set.extend(
                    values
                        .iter()
                        .filter_map(|value| value.as_str())
                        .map(str::to_string),
                );
            }
            if let Some(values) = payload.get("stop").and_then(|value| value.as_array()) {
                stop_set.extend(
                    values
                        .iter()
                        .filter_map(|value| value.as_str())
                        .map(str::to_string),
                );
            }
        }
    }

    skip_set.extend(skip.iter().map(|value| value.to_string()));
    stop_set.extend(stop.iter().map(|value| value.to_string()));

    let payload = serde_json::json!({
        "skip": skip_set.into_iter().collect::<Vec<_>>(),
        "stop": stop_set.into_iter().collect::<Vec<_>>(),
    });
    let temp_path = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(CONTROL_FILE)
    ));
    std::fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|error| format!("Could not create output folder for file control: {error}"))?;
    std::fs::write(
        &temp_path,
        serde_json::to_string_pretty(&payload).unwrap_or_default(),
    )
    .map_err(|error| format!("Could not write file control state: {error}"))?;
    std::fs::rename(&temp_path, path)
        .map_err(|error| format!("Could not save file control state: {error}"))?;
    Ok(())
}

fn cleanup_slide_temp_dirs() -> usize {
    let temp_dir = env::temp_dir();
    let Ok(entries) = std::fs::read_dir(temp_dir) else {
        return 0;
    };

    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_type()
                .map(|file_type| file_type.is_dir())
                .unwrap_or(false)
        })
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| name.starts_with(SLIDE_TEMP_PREFIX))
                .unwrap_or(false)
        })
        .filter(|entry| {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                return true;
            };
            slide_temp_pid(name)
                .map(|pid| !pid_is_running(pid))
                .unwrap_or(true)
        })
        .map(|entry| remove_temp_path(&entry.path()))
        .sum()
}

fn cleanup_output_temp_files(output_dir: &Path) -> usize {
    if !output_dir.exists() {
        return 0;
    }

    let mut deleted = cleanup_output_temp_files_recursive(output_dir);
    let lock_path = output_dir.join(OUTPUT_LOCK_FILE);
    if lock_path.exists() && !lock_owner_is_running(&lock_path) {
        deleted += remove_temp_path(&lock_path);
    }
    deleted
}

fn cleanup_output_temp_files_recursive(folder: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return 0;
    };

    let mut deleted = 0;
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            deleted += cleanup_output_temp_files_recursive(&path);
        } else if file_type.is_file() && is_output_temp_file(&path) {
            deleted += remove_temp_path(&path);
        }
    }
    deleted
}

fn is_output_temp_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    name == NORMALIZED_WORK_FILE
        || name == TRANSCRIPTION_AUDIO_FILE
        || name.ends_with(FFMPEG_TEMP_SUFFIX)
        || ATOMIC_TEMP_FILES.contains(&name)
}

fn remove_temp_path(path: &Path) -> usize {
    let result = if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    };
    if result.is_ok() {
        1
    } else {
        0
    }
}

fn lock_owner_is_running(lock_path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(lock_path) else {
        return false;
    };
    let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    payload
        .get("pid")
        .and_then(|value| value.as_u64())
        .and_then(|pid| u32::try_from(pid).ok())
        .map(pid_is_running)
        .unwrap_or(false)
}

fn slide_temp_pid(name: &str) -> Option<u32> {
    name.strip_prefix(SLIDE_TEMP_PREFIX)?
        .split_once("-")?
        .0
        .parse()
        .ok()
}

#[cfg(unix)]
fn pid_is_running(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(unix)]
fn process_group_is_running(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &format!("-{pid}")])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn pid_is_running(_pid: u32) -> bool {
    false
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = AppState::default();
    let window_close_active_process = Arc::clone(&state.active_process);
    let app_exit_active_process = Arc::clone(&state.active_process);

    tauri::Builder::default()
        .manage(state)
        .on_window_event(move |_window, event| {
            if matches!(
                event,
                tauri::WindowEvent::CloseRequested { .. } | tauri::WindowEvent::Destroyed
            ) {
                terminate_active_process_for_shutdown(&window_close_active_process);
            }
        })
        .setup(|_app| {
            let _ = cleanup_temp_files_impl(None);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            choose_folder,
            default_output_dir,
            scan_folder,
            open_path,
            cleanup_temp_files,
            save_api_key,
            has_api_key,
            system_metrics,
            dependency_status,
            setup_dependencies,
            process_batch,
            cancel_batch,
            update_file_control
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |_app_handle, event| {
            if matches!(
                event,
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
            ) {
                terminate_active_process_for_shutdown(&app_exit_active_process);
            }
        });
}
