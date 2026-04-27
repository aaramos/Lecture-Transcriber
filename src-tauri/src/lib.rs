use serde::{Deserialize, Serialize};
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::Emitter;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

const EVENT_PREFIX: &str = "__LECTURE_PROCESSOR_EVENT__ ";
const SLIDE_TEMP_PREFIX: &str = "lecture-slides-";
const OUTPUT_LOCK_FILE: &str = ".lecture_processor.lock";
const NORMALIZED_WORK_FILE: &str = ".normalized_work.mp4";
const FFMPEG_TEMP_SUFFIX: &str = ".ffmpeg.tmp";
const ATOMIC_TEMP_FILES: [&str; 5] = [
    ".batch_error.txt.tmp",
    ".batch_summary.txt.tmp",
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
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FolderScan {
    mov_count: usize,
    mov_files: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessRequest {
    input_dir: String,
    output_dir: String,
    recording_speed: String,
    confirm_normalization: bool,
    concurrent_files: u8,
    save_normalized_video: bool,
    audio_quality: String,
    transcription_engine: String,
    whisper_model: String,
    slide_sensitivity: String,
    min_duration: f64,
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

#[tauri::command]
fn choose_folder() -> Result<Option<String>, String> {
    Ok(rfd::FileDialog::new()
        .pick_folder()
        .map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
fn default_output_dir(input_dir: String) -> Result<String, String> {
    let input = PathBuf::from(input_dir);
    let name = input
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Invalid input folder.".to_string())?;
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    Ok(parent
        .join(format!("{name}_processed"))
        .to_string_lossy()
        .to_string())
}

#[tauri::command]
fn scan_folder(input_dir: String) -> Result<FolderScan, String> {
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
    Ok(FolderScan {
        mov_count: mov_files.len(),
        mov_files,
    })
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
    let process_to_cancel = {
        let mut active = state
            .active_process
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(process) = active.as_mut() {
            let previous_cancelled = process.cancelled;
            process.cancelled = true;
            Some((process.pid, previous_cancelled))
        } else {
            None
        }
    };

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

fn run_process_batch(
    app: tauri::AppHandle,
    active_process: Arc<Mutex<Option<ActiveProcess>>>,
    request: ProcessRequest,
) -> Result<ProcessResponse, String> {
    if request.recording_speed == "2x" && !request.confirm_normalization {
        return Err("2x normalization requires confirmation.".to_string());
    }
    if request.concurrent_files == 0 || request.concurrent_files > 8 {
        return Err("Concurrent files must be between 1 and 8.".to_string());
    }
    if request.output_dir.trim().is_empty() {
        return Err("Choose an output folder before starting.".to_string());
    }
    if active_process
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .is_some()
    {
        return Err("A batch is already running.".to_string());
    }

    let project_root = find_project_root()?;
    let cli = project_root.join(".venv/bin/lecture-processor");
    if !cli.exists() {
        return Err(format!(
            "Processor CLI was not found at {}. Run . ./scripts/dev-env.sh and install dependencies.",
            cli.display()
        ));
    }

    let mut args = vec![
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
        "--whisper-model".to_string(),
        request.whisper_model.clone(),
        "--slide-sensitivity".to_string(),
        request.slide_sensitivity.clone(),
        "--min-duration".to_string(),
        request.min_duration.to_string(),
        "--json-events".to_string(),
    ];

    if is_apple_silicon() {
        args.push("--apple-silicon".to_string());
        args.push("--slide-backend".to_string());
        args.push("ffmpeg".to_string());
    }

    let whisper_cpp_model_dir = env::var("LECTURE_PROCESSOR_WHISPER_CPP_MODEL_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| default_whisper_cpp_model_dir(&project_root, &request.whisper_model));
    if let Some(model_dir) = whisper_cpp_model_dir {
        args.push("--whisper-cpp-model-dir".to_string());
        args.push(model_dir);
    }

    if env::var("LECTURE_PROCESSOR_REQUIRE_WHISPER_CPP_COREML")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        args.push("--require-whisper-cpp-coreml".to_string());
    }

    if request.confirm_normalization {
        args.push("--confirm-normalization".to_string());
    }

    if !request.save_normalized_video {
        args.push("--no-save-normalized-video".to_string());
    }

    let path = tool_path(&project_root);
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
        output_dir: request.output_dir,
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
    if status.success() {
        return Ok(());
    }
    Err("Could not request cancellation for the running batch.".to_string())
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

fn find_project_root() -> Result<PathBuf, String> {
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

fn tool_path(project_root: &Path) -> String {
    let mut paths = vec![
        project_root.join(".tools/darwin_arm64"),
        project_root.join(".venv/bin"),
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

#[cfg(not(unix))]
fn pid_is_running(_pid: u32) -> bool {
    false
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
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
            process_batch,
            cancel_batch
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
