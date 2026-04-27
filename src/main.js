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
  progressTotal: 0,
  progressDone: 0,
  files: new Map(),
};

const elements = {
  dropZone: document.querySelector("#dropZone"),
  chooseFolderButton: document.querySelector("#chooseFolderButton"),
  clearFolderButton: document.querySelector("#clearFolderButton"),
  chooseOutputButton: document.querySelector("#chooseOutputButton"),
  startButton: document.querySelector("#startButton"),
  openOutputButton: document.querySelector("#openOutputButton"),
  settingsButton: document.querySelector("#settingsButton"),
  settingsDialog: document.querySelector("#settingsDialog"),
  confirmDialog: document.querySelector("#confirmDialog"),
  confirmCheckbox: document.querySelector("#confirmCheckbox"),
  confirmContinue: document.querySelector("#confirmContinue"),
  normalizationWarning: document.querySelector("#normalizationWarning"),
  folderTitle: document.querySelector("#folderTitle"),
  folderSub: document.querySelector("#folderSub"),
  folderError: document.querySelector("#folderError"),
  outputPath: document.querySelector("#outputPath"),
  runTitle: document.querySelector("#runTitle"),
  statusPill: document.querySelector("#statusPill"),
  progressBar: document.querySelector("#progressBar"),
  logOutput: document.querySelector("#logOutput"),
  fileList: document.querySelector("#fileList"),
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

window.__TAURI__?.event?.listen?.("processor-event", (event) => handleProcessorEvent(event.payload));
setupDragAndDrop();

document.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (event.metaKey && key === ",") {
    event.preventDefault();
    showDialog(elements.settingsDialog);
  } else if (event.metaKey && key === "r") {
    event.preventDefault();
    if (elements.settingsDialog.open || elements.confirmDialog.open) return;
    startBatch();
  } else if (event.key === "Escape" && !state.running && !elements.settingsDialog.open && !elements.confirmDialog.open) {
    clearFolder();
  }
});

elements.dropZone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  chooseInputFolder();
});
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    chooseInputFolder();
  }
});
elements.chooseFolderButton.addEventListener("click", chooseInputFolder);
elements.chooseOutputButton.addEventListener("click", chooseOutputFolder);
elements.clearFolderButton.addEventListener("click", clearFolder);
elements.startButton.addEventListener("click", startBatch);
elements.openOutputButton.addEventListener("click", openOutput);
elements.settingsButton.addEventListener("click", () => showDialog(elements.settingsDialog));
elements.confirmCheckbox.addEventListener("change", () => {
  elements.confirmContinue.disabled = !elements.confirmCheckbox.checked;
});
elements.concurrentFiles.addEventListener("input", validateSettings);

elements.speedSegments.forEach((button) => {
  button.addEventListener("click", () => {
    state.recordingSpeed = button.dataset.speed;
    elements.speedSegments.forEach((item) => item.classList.toggle("active", item === button));
    renderNormalizationWarning();
  });
});

async function chooseInputFolder() {
  const folder = await invoke("choose_folder");
  if (!folder) return;
  await selectInputFolder(folder);
}

async function selectInputFolder(folder) {
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
    const confirmed = await confirmNormalization();
    if (!confirmed) return;
  }

  const concurrentFiles = validateSettings();
  if (!concurrentFiles) return;

  setRunning(true);
  resetResults();
  state.files.clear();
  renderFileList();
  elements.logOutput.textContent = "Processing batch...\n";

  const request = {
    inputDir: state.inputDir,
    outputDir: state.outputDir,
    recordingSpeed: state.recordingSpeed,
    confirmNormalization: state.recordingSpeed === "2x",
    concurrentFiles,
    saveNormalizedVideo: elements.saveNormalized.checked,
    audioQuality: elements.audioQuality.value,
    transcriptionEngine: elements.transcriptionEngine.value,
    whisperModel: elements.whisperModel.value,
    slideSensitivity: elements.slideSensitivity.value,
    minDuration: 60,
  };

  try {
    const result = await invoke("process_batch", { request });
    const logText = [result.stdout, result.stderr].filter(Boolean).join("\n\n").trim();
    state.lastOutputDir = result.outputDir;
    appendLog(logText || "Batch finished.");
    elements.runTitle.textContent = result.exitCode === 0 ? "Batch finished" : "Batch finished with issues";
    elements.statusPill.textContent = result.exitCode === 0 ? "Complete" : "Review";
    elements.statusPill.className = result.exitCode === 0 ? "status-pill complete" : "status-pill warning";
    renderSummary(logText);
    elements.openOutputButton.disabled = false;
  } catch (error) {
    elements.runTitle.textContent = "Batch failed";
    elements.statusPill.textContent = "Failed";
    elements.statusPill.className = "status-pill failed";
    appendLog(String(error));
  } finally {
    setRunning(false);
  }
}

async function openOutput() {
  const path = state.lastOutputDir || state.outputDir;
  if (!path) return;
  try {
    await invoke("open_path", { path });
  } catch (error) {
    appendLog(`Could not open output folder: ${error}`);
  }
}

function render() {
  const hasFolder = Boolean(state.inputDir);
  elements.folderTitle.textContent = hasFolder ? basename(state.inputDir) : "Choose lecture folder";
  elements.folderSub.textContent = hasFolder
    ? `${state.inputDir} · ${state.fileCount} .mov file${state.fileCount === 1 ? "" : "s"}`
    : "No folder selected";
  elements.outputPath.textContent = state.outputDir || "-";
  elements.clearFolderButton.classList.toggle("hidden", !hasFolder);
  elements.startButton.disabled = state.running || !hasFolder || state.fileCount === 0;
  renderNormalizationWarning();
}

function setRunning(running) {
  state.running = running;
  elements.startButton.disabled = running || !state.inputDir || state.fileCount === 0;
  elements.chooseFolderButton.disabled = running;
  elements.clearFolderButton.disabled = running;
  elements.chooseOutputButton.disabled = running;
  elements.progressBar.classList.toggle("running", running);
  if (running) {
    elements.runTitle.textContent = "Processing";
    elements.statusPill.textContent = "Running";
    elements.statusPill.className = "status-pill running";
    setProgress(0, 0);
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
  state.progressTotal = 0;
  state.progressDone = 0;
  setProgress(0, 0);
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

function confirmNormalization() {
  return new Promise((resolve) => {
    elements.confirmCheckbox.checked = false;
    elements.confirmContinue.disabled = true;

    const onClose = () => {
      resolve(elements.confirmDialog.returnValue === "confirm" && elements.confirmCheckbox.checked);
    };

    elements.confirmDialog.addEventListener("close", onClose, { once: true });
    showDialog(elements.confirmDialog);
  });
}

function validateSettings() {
  const value = Number.parseInt(elements.concurrentFiles.value, 10);
  const valid = Number.isInteger(value) && value >= 1 && value <= 8;
  elements.concurrentFiles.classList.toggle("invalid", !valid);
  if (!valid) {
    showDialog(elements.settingsDialog);
    elements.concurrentFiles.focus();
    appendLog("Concurrent files must be between 1 and 8.");
    return null;
  }
  return value;
}

function handleProcessorEvent(event) {
  if (!event || typeof event !== "object") return;

  if (event.kind === "batch_started") {
    state.progressTotal = Number(event.attempted || 0);
    state.progressDone = 0;
    setProgress(0, state.progressTotal);
    appendLog(`Found ${state.progressTotal} file${state.progressTotal === 1 ? "" : "s"}.`);
  } else if (event.kind === "file_started") {
    updateFile(event.source, { status: "running", detail: "Starting" });
    appendLog(`${event.source}: starting`);
  } else if (event.kind === "step_started") {
    updateFile(event.source, { status: "running", detail: event.step });
  } else if (event.kind === "step_finished") {
    updateFile(event.source, { detail: `${event.step} done (${event.elapsed_seconds}s)` });
  } else if (event.kind === "file_finished") {
    state.progressDone = Number(event.attempted || state.progressDone + 1);
    updateFile(event.source, {
      status: event.status,
      detail: event.status === "completed" ? `${event.word_count} words, ${event.slide_count} slides` : event.message,
    });
    updateCounts(event);
    setProgress(state.progressDone, state.progressTotal);
    appendLog(`${event.source}: ${event.status}`);
  } else if (event.kind === "batch_finished") {
    updateCounts(event);
    setProgress(Number(event.attempted || state.progressDone), Number(event.attempted || state.progressTotal));
  }
}

function updateFile(source, patch) {
  if (!source) return;
  const previous = state.files.get(source) || { status: "queued", detail: "Queued" };
  state.files.set(source, { ...previous, ...patch });
  renderFileList();
}

function renderFileList() {
  if (state.files.size === 0) {
    elements.fileList.innerHTML = `<div class="file-placeholder">${
      state.running ? "Waiting for processor events." : "No files running."
    }</div>`;
    return;
  }

  elements.fileList.innerHTML = [...state.files.entries()]
    .map(([name, file]) => {
      const status = escapeHtml(file.status || "queued");
      const detail = escapeHtml(file.detail || "");
      return `
        <div class="file-row ${status}">
          <span class="file-name">${escapeHtml(name)}</span>
          <span class="file-detail">${detail}</span>
        </div>
      `;
    })
    .join("");
}

function updateCounts(event) {
  if (event.attempted !== undefined) elements.attemptedCount.textContent = String(event.attempted);
  if (event.completed !== undefined) elements.completedCount.textContent = String(event.completed);
  if (event.failed !== undefined) elements.failedCount.textContent = String(event.failed);
  if (event.skipped !== undefined) elements.skippedCount.textContent = String(event.skipped);
}

function setProgress(done, total) {
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;
  elements.progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

function appendLog(line) {
  if (!line) return;
  const current = elements.logOutput.textContent;
  const prefix = current === "Waiting for a batch." ? "" : current.replace(/\s*$/, "\n");
  elements.logOutput.textContent = `${prefix}${line}`;
  elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
}

function renderNormalizationWarning() {
  elements.normalizationWarning.classList.toggle("hidden", state.recordingSpeed !== "2x");
}

function setupDragAndDrop() {
  const webview = window.__TAURI__?.webview?.getCurrentWebview?.();
  webview
    ?.onDragDropEvent?.((event) => {
      if (event.payload?.type === "drop") {
        selectInputFolder(event.payload.paths?.[0]);
      }
    })
    ?.catch?.(() => {});

  elements.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });

  elements.dropZone.addEventListener("dragleave", () => {
    elements.dropZone.classList.remove("dragging");
  });

  elements.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
    const path = event.dataTransfer?.files?.[0]?.path;
    if (path) {
      selectInputFolder(path);
    } else {
      setFolderError("Use Browse if the dropped folder path is unavailable.");
    }
  });
}

function showDialog(dialog) {
  if (!dialog.open) dialog.showModal();
}

function basename(path) {
  return path.split(/[\\/]/).filter(Boolean).pop() || path;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return entities[char];
  });
}

render();
