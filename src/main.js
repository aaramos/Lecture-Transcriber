const invoke = window.__TAURI__?.core?.invoke;

if (!invoke) {
  throw new Error("Tauri API is not available. Start the app with `cargo tauri dev`.");
}

const state = {
  inputDir: "",
  outputDir: "",
  fileCount: 0,
  recordingSpeed: "1x",
  running: false,
  lastOutputDir: "",
};

const elements = {
  chooseFolderButton: document.querySelector("#chooseFolderButton"),
  clearFolderButton: document.querySelector("#clearFolderButton"),
  chooseOutputButton: document.querySelector("#chooseOutputButton"),
  startButton: document.querySelector("#startButton"),
  openOutputButton: document.querySelector("#openOutputButton"),
  settingsButton: document.querySelector("#settingsButton"),
  settingsDialog: document.querySelector("#settingsDialog"),
  folderTitle: document.querySelector("#folderTitle"),
  folderSub: document.querySelector("#folderSub"),
  folderError: document.querySelector("#folderError"),
  outputPath: document.querySelector("#outputPath"),
  runTitle: document.querySelector("#runTitle"),
  statusPill: document.querySelector("#statusPill"),
  progressBar: document.querySelector("#progressBar"),
  logOutput: document.querySelector("#logOutput"),
  attemptedCount: document.querySelector("#attemptedCount"),
  completedCount: document.querySelector("#completedCount"),
  failedCount: document.querySelector("#failedCount"),
  skippedCount: document.querySelector("#skippedCount"),
  audioQuality: document.querySelector("#audioQuality"),
  transcriptionEngine: document.querySelector("#transcriptionEngine"),
  whisperModel: document.querySelector("#whisperModel"),
  slideSensitivity: document.querySelector("#slideSensitivity"),
  concurrentFiles: document.querySelector("#concurrentFiles"),
  saveNormalized: document.querySelector("#saveNormalized"),
  speedSegments: [...document.querySelectorAll(".segment")],
};

elements.chooseFolderButton.addEventListener("click", chooseInputFolder);
elements.chooseOutputButton.addEventListener("click", chooseOutputFolder);
elements.clearFolderButton.addEventListener("click", clearFolder);
elements.startButton.addEventListener("click", startBatch);
elements.openOutputButton.addEventListener("click", openOutput);
elements.settingsButton.addEventListener("click", () => elements.settingsDialog.showModal());

elements.speedSegments.forEach((button) => {
  button.addEventListener("click", () => {
    state.recordingSpeed = button.dataset.speed;
    elements.speedSegments.forEach((item) => item.classList.toggle("active", item === button));
  });
});

async function chooseInputFolder() {
  const folder = await invoke("choose_folder");
  if (!folder) return;

  state.inputDir = folder;
  state.outputDir = await invoke("default_output_dir", { inputDir: folder });
  await scanFolder();
  render();
}

async function chooseOutputFolder() {
  const folder = await invoke("choose_folder");
  if (!folder) return;
  state.outputDir = folder;
  render();
}

async function scanFolder() {
  try {
    const scan = await invoke("scan_folder", { inputDir: state.inputDir });
    state.fileCount = scan.movCount;
    setFolderError(scan.movCount === 0 ? "No .mov files found. Try a different folder." : "");
  } catch (error) {
    state.fileCount = 0;
    setFolderError(String(error));
  }
}

function clearFolder() {
  state.inputDir = "";
  state.outputDir = "";
  state.fileCount = 0;
  setFolderError("");
  render();
}

async function startBatch() {
  if (state.running || !state.inputDir || state.fileCount === 0) return;

  if (state.recordingSpeed === "2x") {
    const confirmed = window.confirm(
      "Confirm that every eligible file in this batch was recorded at 2x speed. Running this on 1x recordings will produce half-speed output."
    );
    if (!confirmed) return;
  }

  setRunning(true);
  resetResults();
  elements.logOutput.textContent = "Processing batch...";

  const request = {
    inputDir: state.inputDir,
    outputDir: state.outputDir,
    recordingSpeed: state.recordingSpeed,
    confirmNormalization: state.recordingSpeed === "2x",
    concurrentFiles: Number(elements.concurrentFiles.value),
    saveNormalizedVideo: elements.saveNormalized.checked,
    audioQuality: elements.audioQuality.value,
    transcriptionEngine: elements.transcriptionEngine.value,
    whisperModel: elements.whisperModel.value,
    slideSensitivity: elements.slideSensitivity.value,
    minDuration: 60,
  };

  try {
    const result = await invoke("process_batch", { request });
    const logText = [result.stdout, result.stderr].filter(Boolean).join("\n\n");
    state.lastOutputDir = result.outputDir;
    elements.logOutput.textContent = logText || "Batch finished.";
    elements.runTitle.textContent = result.exitCode === 0 ? "Batch finished" : "Batch finished with issues";
    elements.statusPill.textContent = result.exitCode === 0 ? "Complete" : "Review";
    elements.statusPill.className = result.exitCode === 0 ? "status-pill complete" : "status-pill warning";
    renderSummary(logText);
    elements.openOutputButton.disabled = false;
  } catch (error) {
    elements.runTitle.textContent = "Batch failed";
    elements.statusPill.textContent = "Failed";
    elements.statusPill.className = "status-pill failed";
    elements.logOutput.textContent = String(error);
  } finally {
    setRunning(false);
  }
}

async function openOutput() {
  const path = state.lastOutputDir || state.outputDir;
  if (!path) return;
  await invoke("open_path", { path });
}

function render() {
  const hasFolder = Boolean(state.inputDir);
  elements.folderTitle.textContent = hasFolder ? basename(state.inputDir) : "Choose lecture folder";
  elements.folderSub.textContent = hasFolder
    ? `${state.inputDir} · ${state.fileCount} .mov file${state.fileCount === 1 ? "" : "s"}`
    : "No folder selected";
  elements.outputPath.textContent = state.outputDir || "—";
  elements.clearFolderButton.classList.toggle("hidden", !hasFolder);
  elements.startButton.disabled = state.running || !hasFolder || state.fileCount === 0;
}

function setRunning(running) {
  state.running = running;
  elements.startButton.disabled = running || !state.inputDir || state.fileCount === 0;
  elements.chooseFolderButton.disabled = running;
  elements.chooseOutputButton.disabled = running;
  elements.progressBar.classList.toggle("running", running);
  if (running) {
    elements.runTitle.textContent = "Processing";
    elements.statusPill.textContent = "Running";
    elements.statusPill.className = "status-pill running";
  }
}

function setFolderError(message) {
  elements.folderError.textContent = message;
  elements.folderError.classList.toggle("hidden", !message);
}

function resetResults() {
  elements.attemptedCount.textContent = "0";
  elements.completedCount.textContent = "0";
  elements.failedCount.textContent = "0";
  elements.skippedCount.textContent = "0";
  elements.openOutputButton.disabled = true;
}

function renderSummary(stdout) {
  const countFor = (label) => {
    const match = stdout.match(new RegExp(`${label}:\\s+(\\d+)`));
    return match ? match[1] : "0";
  };
  elements.attemptedCount.textContent = countFor("Attempted");
  elements.completedCount.textContent = countFor("Completed");
  elements.failedCount.textContent = countFor("Failed");
  elements.skippedCount.textContent = countFor("Skipped");
}

function basename(path) {
  return path.split(/[\\/]/).filter(Boolean).pop() || path;
}

render();
