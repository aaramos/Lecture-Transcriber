use serde::{Deserialize, Serialize};
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::Emitter;

const EVENT_PREFIX: &str = "__LECTURE_PROCESSOR_EVENT__ ";

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
async fn process_batch(
    app: tauri::AppHandle,
    request: ProcessRequest,
) -> Result<ProcessResponse, String> {
    tauri::async_runtime::spawn_blocking(move || run_process_batch(app, request))
        .await
        .map_err(|error| format!("Processor task failed: {error}"))?
}

fn run_process_batch(
    app: tauri::AppHandle,
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

    if let Ok(model_dir) = env::var("LECTURE_PROCESSOR_WHISPER_CPP_MODEL_DIR") {
        if !model_dir.trim().is_empty() {
            args.push("--whisper-cpp-model-dir".to_string());
            args.push(model_dir);
        }
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
    let mut child = Command::new(cli)
        .args(args)
        .current_dir(&project_root)
        .env("PATH", path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not start processor: {error}"))?;

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

    let status = child
        .wait()
        .map_err(|error| format!("Processor failed while running: {error}"))?;
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
        exit_code: status.code().unwrap_or(1),
        stdout: stdout_text,
        stderr: stderr_text,
        output_dir: request.output_dir,
    })
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            choose_folder,
            default_output_dir,
            scan_folder,
            open_path,
            process_batch
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
