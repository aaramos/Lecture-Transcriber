use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FolderScan {
    mov_count: usize,
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
    let mov_count = entries
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
        .count();
    Ok(FolderScan { mov_count })
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let status = Command::new("open")
        .arg(path)
        .status()
        .map_err(|error| format!("Could not open output folder: {error}"))?;
    if !status.success() {
        return Err("Could not open output folder.".to_string());
    }
    Ok(())
}

#[tauri::command]
fn process_batch(request: ProcessRequest) -> Result<ProcessResponse, String> {
    if request.recording_speed == "2x" && !request.confirm_normalization {
        return Err("2x normalization requires confirmation.".to_string());
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
    ];

    if request.confirm_normalization {
        args.push("--confirm-normalization".to_string());
    }

    if !request.save_normalized_video {
        args.push("--no-save-normalized-video".to_string());
    }

    let path = tool_path(&project_root);
    let output = Command::new(cli)
        .args(args)
        .current_dir(&project_root)
        .env("PATH", path)
        .output()
        .map_err(|error| format!("Could not start processor: {error}"))?;

    Ok(ProcessResponse {
        exit_code: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        output_dir: request.output_dir,
    })
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
