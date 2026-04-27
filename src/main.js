const invoke = window.__TAURI__?.core?.invoke;

if (!invoke) {
  throw new Error("Tauri API is not available. Start the app with `cargo tauri dev`.");
}

const STEP_LABELS = {
  Probe: "Checking video",
  Normalize: "Normalizing",
  Transcribe: "Transcribing",
  Slides: "Extracting slides",
};

const STEP_START_PROGRESS = {
  Probe: 8,
  Normalize: 22,
  Transcribe: 48,
  Slides: 82,
};

const STEP_DONE_PROGRESS = {
  Probe: 18,
  Normalize: 42,
  Transcribe: 78,
  Slides: 96,
};

const SETTINGS_STORAGE_KEY = "lectureProcessor.settings.v1";
const LAST_OUTPUT_STORAGE_KEY = "lectureProcessor.lastOutputDir.v1";
const DEFAULT_SETTINGS = Object.freeze({
  recordingSpeed: "1x",
  audioQuality: "fast",
  transcriptionEngine: "auto",
  whisperModel: "large-v3",
  slideSensitivity: "medium",
  concurrentFiles: 4,
  saveNormalized: true,
});

const state = {
  inputDir: "",
  outputDir: "",
  fileCount: 0,
  fileNames: [],
  recordingSpeed: DEFAULT_SETTINGS.recordingSpeed,
  running: false,
  cancelRequested: false,
  processorStarted: false,
  lastOutputDir: "",
  progressTotal: 0,
  progressDone: 0,
  runStartedAt: 0,
  elapsedTimer: null,
  finishedFileReports: 0,
  files: new Map(),
  fileOrder: [],
};

const elements = {
  dropZone: document.querySelector("#dropZone"),
  chooseFolderButton: document.querySelector("#chooseFolderButton"),
  clearFolderButton: document.querySelector("#clearFolderButton"),
  chooseOutputButton: document.querySelector("#chooseOutputButton"),
  startButton: document.querySelector("#startButton"),
  cancelRunButton: document.querySelector("#cancelRunButton"),
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
  runMeta: document.querySelector("#runMeta"),
  progressPercent: document.querySelector("#progressPercent"),
  statusPill: document.querySelector("#statusPill"),
  elapsedTime: document.querySelector("#elapsedTime"),
  progressBar: document.querySelector("#progressBar"),
  activeVideos: document.querySelector("#activeVideos"),
  queueList: document.querySelector("#queueList"),
  activeSummary: document.querySelector("#activeSummary"),
  queueSummary: document.querySelector("#queueSummary"),
  completedCount: document.querySelector("#completedCount"),
  processingCount: document.querySelector("#processingCount"),
  waitingCount: document.querySelector("#waitingCount"),
  failedCount: document.querySelector("#failedCount"),
  skippedCount: document.querySelector("#skippedCount"),
  canceledCount: document.querySelector("#canceledCount"),
  audioQuality: document.querySelector("#audioQuality"),
  transcriptionEngine: document.querySelector("#transcriptionEngine"),
  whisperModel: document.querySelector("#whisperModel"),
  slideSensitivity: document.querySelector("#slideSensitivity"),
  concurrentFiles: document.querySelector("#concurrentFiles"),
  saveNormalized: document.querySelector("#saveNormalized"),
  restoreDefaultsButton: document.querySelector("#restoreDefaultsButton"),
  speedSegments: [...document.querySelectorAll(".segment")],
};

window.__TAURI__?.event?.listen?.("processor-event", (event) => handleProcessorEvent(event.payload));
loadPersistedSettings();
cleanupTempFilesAtLaunch();
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
elements.cancelRunButton.addEventListener("click", cancelBatch);
elements.openOutputButton.addEventListener("click", openOutput);
elements.settingsButton.addEventListener("click", () => showDialog(elements.settingsDialog));
elements.restoreDefaultsButton.addEventListener("click", restoreDefaultSettings);
elements.confirmCheckbox.addEventListener("change", () => {
  elements.confirmContinue.disabled = !elements.confirmCheckbox.checked;
});
elements.concurrentFiles.addEventListener("input", () => {
  validateSettings({ showDialogOnError: false });
  saveCurrentSettings();
});

[
  elements.audioQuality,
  elements.transcriptionEngine,
  elements.whisperModel,
  elements.slideSensitivity,
  elements.saveNormalized,
].forEach((element) => {
  element.addEventListener("change", saveCurrentSettings);
});

elements.speedSegments.forEach((button) => {
  button.addEventListener("click", () => {
    state.recordingSpeed = button.dataset.speed;
    elements.speedSegments.forEach((item) => item.classList.toggle("active", item === button));
    saveCurrentSettings();
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
  rememberOutputDir(state.outputDir);
  await scanFolder();
  render();
}

async function chooseOutputFolder() {
  const folder = await invoke("choose_folder");
  if (!folder) return;
  state.outputDir = folder;
  rememberOutputDir(state.outputDir);
  render();
}

async function scanFolder() {
  try {
    const scan = await invoke("scan_folder", { inputDir: state.inputDir });
    state.fileCount = scan.movCount;
    state.fileNames = scan.movFiles || [];
    state.files.clear();
    state.fileOrder = [];
    setFolderError(scan.movCount === 0 ? "No .mov files found. Try a different folder." : "");
  } catch (error) {
    state.fileCount = 0;
    state.fileNames = [];
    state.files.clear();
    state.fileOrder = [];
    setFolderError(String(error));
  }
}

function clearFolder() {
  state.inputDir = "";
  state.outputDir = "";
  state.fileCount = 0;
  state.fileNames = [];
  state.files.clear();
  state.fileOrder = [];
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

  resetResults();
  setRunning(true);
  primeQueuedFiles(concurrentFiles);
  renderVideoDashboard();
  await waitForPaint();

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
  rememberOutputDir(request.outputDir);

  try {
    state.processorStarted = true;
    renderRunControls();
    const result = await invoke("process_batch", { request });
    applyProcessorResult(result);
  } catch (error) {
    state.lastOutputDir = state.outputDir;
    markUnfinishedVideosFailed("Processor stopped before reporting this video complete.");
    renderVideoDashboard();
    elements.runTitle.textContent = "Batch failed";
    elements.statusPill.textContent = "Failed";
    elements.statusPill.className = "status-pill failed";
    elements.runMeta.textContent = `${String(error)}. Open the output folder for batch_error.txt.`;
    elements.openOutputButton.disabled = !state.outputDir;
  } finally {
    state.cancelRequested = false;
    state.processorStarted = false;
    setRunning(false);
  }
}

async function cancelBatch() {
  if (!state.running || state.cancelRequested || !state.processorStarted) return;

  state.cancelRequested = true;
  renderRunControls();
  elements.runTitle.textContent = "Canceling batch...";
  elements.statusPill.textContent = "Canceling";
  elements.statusPill.className = "status-pill warning";
  elements.runMeta.textContent = "Stopping active video processing.";
  updateActiveVideosForCancel();

  try {
    const result = await invoke("cancel_batch");
    if (!result.cancelled) {
      state.cancelRequested = false;
      elements.runMeta.textContent = result.message || "No active batch was found to cancel.";
      renderRunControls();
    }
  } catch (error) {
    state.cancelRequested = false;
    elements.runTitle.textContent = "Processing batch";
    elements.statusPill.textContent = "Running";
    elements.statusPill.className = "status-pill running";
    elements.runMeta.textContent = `Could not cancel batch: ${error}`;
    renderRunControls();
  }
}

async function openOutput() {
  const path = state.lastOutputDir || state.outputDir;
  if (!path) return;
  try {
    await invoke("open_path", { path });
  } catch (error) {
    elements.runMeta.textContent = `Could not open output folder: ${error}`;
  }
}

function applyProcessorResult(result) {
  state.lastOutputDir = result.outputDir || state.outputDir;
  rememberOutputDir(state.lastOutputDir);

  let unfinished = 0;
  if (result.cancelled) {
    unfinished = markRemainingVideosCanceled();
  } else {
    unfinished = markUnfinishedVideosFailed("Processor ended before reporting this video complete.");
  }

  renderVideoDashboard();
  const counts = countDashboardEntries(dashboardEntries());
  const exitCode = Number(result.exitCode || 0);
  const hasFailures = counts.failed > 0 || exitCode !== 0;
  const hasSkips = counts.skipped > 0;
  const failedBeforeFileResults = exitCode !== 0 && state.finishedFileReports === 0;
  const allReported =
    counts.processing === 0 &&
    counts.waiting === 0 &&
    counts.canceled === 0 &&
    counts.failed === 0 &&
    counts.completed + counts.skipped === (state.progressTotal || state.fileCount || counts.total);

  if (result.cancelled) {
    elements.runTitle.textContent = "Batch canceled";
    elements.statusPill.textContent = "Canceled";
    elements.statusPill.className = "status-pill warning";
    elements.runMeta.textContent = "Processing was stopped. Completed files remain in the output folder.";
  } else if (exitCode === 0 && allReported && !hasSkips) {
    elements.runTitle.textContent = "Batch finished";
    elements.statusPill.textContent = "Complete";
    elements.statusPill.className = "status-pill complete";
    elements.runMeta.textContent = "All videos finished. Open the output folder for transcripts, videos, and slides.";
  } else {
    elements.runTitle.textContent = failedBeforeFileResults
      ? "Batch failed"
      : hasFailures
        ? "Batch finished with issues"
        : "Batch finished with skips";
    elements.statusPill.textContent = failedBeforeFileResults ? "Failed" : hasFailures ? "Review" : "Complete";
    elements.statusPill.className = failedBeforeFileResults
      ? "status-pill failed"
      : hasFailures
        ? "status-pill warning"
        : "status-pill complete";
    elements.runMeta.textContent = finalRunSummary(counts, unfinished, exitCode);
  }

  elements.openOutputButton.disabled = !state.lastOutputDir;
}

function finalRunSummary(counts, unfinished, exitCode) {
  const parts = [];
  if (counts.completed) parts.push(`${counts.completed} complete`);
  if (counts.failed) parts.push(`${counts.failed} failed`);
  if (counts.skipped) parts.push(`${counts.skipped} skipped`);
  if (counts.canceled) parts.push(`${counts.canceled} canceled`);
  if (unfinished) parts.push(`${unfinished} did not report a final result`);
  if (exitCode !== 0 && counts.failed === 0) parts.push(`processor exited with code ${exitCode}`);
  return `${parts.join(" · ") || "Run needs review"}. Open the output folder for batch_summary.txt or batch_error.txt.`;
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
  if (!state.running) {
    elements.runMeta.textContent = hasFolder
      ? `${state.fileCount} video${state.fileCount === 1 ? "" : "s"} ready`
      : "Select a folder to begin.";
  }
  renderVideoDashboard();
  renderNormalizationWarning();
}

function setRunning(running) {
  state.running = running;
  elements.startButton.disabled = running || !state.inputDir || state.fileCount === 0;
  elements.chooseFolderButton.disabled = running;
  elements.clearFolderButton.disabled = running;
  elements.chooseOutputButton.disabled = running;
  elements.progressBar.classList.toggle("running", running);
  renderRunControls();
  if (running) {
    elements.runTitle.textContent = `Preparing ${state.fileCount} video${state.fileCount === 1 ? "" : "s"}...`;
    elements.statusPill.textContent = "Running";
    elements.statusPill.className = "status-pill running";
    elements.runMeta.textContent = runDescription();
    setProgress(0, 0);
    startElapsedTimer();
  } else {
    stopElapsedTimer();
  }
  renderVideoDashboard();
}

function setFolderError(message) {
  elements.folderError.textContent = message;
  elements.folderError.classList.toggle("hidden", !message);
}

function resetResults() {
  elements.completedCount.textContent = "0";
  elements.processingCount.textContent = "0";
  elements.waitingCount.textContent = String(state.fileCount || 0);
  elements.failedCount.textContent = "0";
  elements.skippedCount.textContent = "0";
  elements.canceledCount.textContent = "0";
  elements.openOutputButton.disabled = true;
  state.progressTotal = 0;
  state.progressDone = 0;
  state.cancelRequested = false;
  state.processorStarted = false;
  state.finishedFileReports = 0;
  state.files.clear();
  state.fileOrder = [];
  setProgress(0, 0);
  renderVideoDashboard();
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

function validateSettings(options = {}) {
  const { showDialogOnError = true } = options;
  const value = Number.parseInt(elements.concurrentFiles.value, 10);
  const valid = Number.isInteger(value) && value >= 1 && value <= 8;
  elements.concurrentFiles.classList.toggle("invalid", !valid);
  if (!valid) {
    if (showDialogOnError) {
      showDialog(elements.settingsDialog);
    }
    elements.concurrentFiles.focus();
    elements.runMeta.textContent = "Concurrent files must be between 1 and 8.";
    return null;
  }
  return value;
}

function handleProcessorEvent(event) {
  if (!event || typeof event !== "object") return;

  if (event.kind === "batch_started") {
    state.processorStarted = true;
    state.progressTotal = Number(event.attempted || 0);
    state.progressDone = 0;
    setProgress(0, state.progressTotal);
    elements.runTitle.textContent = `Processing ${basename(state.inputDir)}`;
    elements.runMeta.textContent = runDescription();
    renderRunControls();
    renderVideoDashboard();
  } else if (event.kind === "file_started") {
    updateFile(event.source, {
      status: "running",
      stage: "Preparing",
      detail: "Preparing video",
      progress: 5,
    });
  } else if (event.kind === "step_started") {
    updateFile(event.source, {
      status: "running",
      stage: STEP_LABELS[event.step] || event.step,
      detail: "Working",
      progress: STEP_START_PROGRESS[event.step] || 12,
    });
  } else if (event.kind === "step_finished") {
    updateFile(event.source, {
      stage: `${STEP_LABELS[event.step] || event.step} complete`,
      detail: `${Number(event.elapsed_seconds || 0).toFixed(1)}s`,
      progress: STEP_DONE_PROGRESS[event.step] || 20,
    });
  } else if (event.kind === "file_finished") {
    state.finishedFileReports += 1;
    state.progressDone = Number(event.attempted || state.progressDone + 1);
    const previous = state.files.get(event.source) || {};
    const failedStage = previous.stage ? `Failed during ${previous.stage.toLowerCase()}` : "Failed";
    updateFile(event.source, {
      status: event.status,
      stage: statusLabel(event.status),
      detail: fileFinishedDetail(event, failedStage),
      progress: 100,
    });
  } else if (event.kind === "batch_finished") {
    setProgress(Number(event.attempted || state.progressDone), Number(event.attempted || state.progressTotal));
    renderVideoDashboard();
  } else if (event.kind === "batch_failed") {
    state.lastOutputDir = event.output_dir || state.outputDir;
    markUnfinishedVideosFailed(event.message || "Processor stopped before reporting this video complete.");
    elements.runTitle.textContent = "Batch failed";
    elements.statusPill.textContent = "Failed";
    elements.statusPill.className = "status-pill failed";
    elements.runMeta.textContent = `${event.message || "The processor stopped early."} Open the output folder for batch_error.txt.`;
    elements.openOutputButton.disabled = !state.lastOutputDir;
  }
}

function updateFile(source, patch) {
  if (!source) return;
  if (!state.fileOrder.includes(source)) {
    state.fileOrder.push(source);
  }
  const previous = state.files.get(source) || {
    status: "queued",
    stage: "Waiting",
    detail: "Waiting",
    progress: 0,
  };
  state.files.set(source, { ...previous, ...patch });
  renderVideoDashboard();
}

function renderVideoDashboard() {
  const entries = dashboardEntries();
  const activeEntries = entries.filter(([_name, file]) => isActiveStatus(file.status));
  const queueEntries = entries.filter(([_name, file]) => !isActiveStatus(file.status));

  renderCounts(entries);
  renderOverallProgress(entries);
  renderActiveVideos(activeEntries);
  renderQueue(queueEntries);
}

function primeQueuedFiles(concurrentFiles) {
  state.files.clear();
  const names = state.fileNames.length
    ? state.fileNames
    : Array.from({ length: state.fileCount }, (_value, index) => `File ${index + 1}`);

  state.fileOrder = names;
  names.forEach((name, index) => {
    state.files.set(name, {
      status: index < concurrentFiles ? "preparing" : "queued",
      stage: index < concurrentFiles ? "Preparing" : "Waiting",
      detail: index < concurrentFiles ? "Preparing video" : "Waiting",
      progress: index < concurrentFiles ? 3 : 0,
    });
  });

  state.progressTotal = names.length;
  elements.runMeta.textContent = `Processing up to ${Math.min(concurrentFiles, names.length)} video${
    Math.min(concurrentFiles, names.length) === 1 ? "" : "s"
  } at once.`;
  renderVideoDashboard();
}

function dashboardEntries() {
  if (state.fileOrder.length > 0) {
    return state.fileOrder.map((name) => [
      name,
      state.files.get(name) || {
        status: "queued",
        stage: "Waiting",
        detail: "Waiting",
        progress: 0,
      },
    ]);
  }

  if (state.fileNames.length > 0) {
    return state.fileNames.map((name) => [
      name,
      {
        status: "queued",
        stage: "Ready",
        detail: "Ready",
        progress: 0,
      },
    ]);
  }

  return [];
}

function renderCounts(entries) {
  const { completed, processing, failed, skipped, canceled, waiting } = countDashboardEntries(entries);

  elements.completedCount.textContent = String(completed);
  elements.processingCount.textContent = String(processing);
  elements.waitingCount.textContent = String(waiting);
  elements.failedCount.textContent = String(failed);
  elements.skippedCount.textContent = String(skipped);
  elements.canceledCount.textContent = String(canceled);
}

function countDashboardEntries(entries) {
  return {
    total: entries.length,
    completed: entries.filter(([_name, file]) => file.status === "completed").length,
    processing: entries.filter(([_name, file]) => isActiveStatus(file.status)).length,
    failed: entries.filter(([_name, file]) => file.status === "failed").length,
    skipped: entries.filter(([_name, file]) => file.status === "skipped").length,
    canceled: entries.filter(([_name, file]) => file.status === "canceled").length,
    waiting: entries.filter(([_name, file]) => file.status === "queued").length,
  };
}

function renderOverallProgress(entries) {
  const total = state.progressTotal || entries.length || state.fileCount || 0;
  const knownProgress = entries.reduce((sum, [_name, file]) => sum + Number(file.progress || 0), 0);
  const percent = total > 0 ? Math.round(knownProgress / total) : 0;
  setProgressPercent(percent);
}

function renderActiveVideos(entries) {
  elements.activeSummary.textContent =
    entries.length === 0
      ? state.running
        ? "Waiting for active videos"
        : "No videos processing"
      : `${entries.length} processing`;

  if (entries.length === 0) {
    elements.activeVideos.innerHTML = `<div class="video-placeholder">${
      state.running ? "Waiting for the next video to start." : "No videos processing."
    }</div>`;
    return;
  }

  elements.activeVideos.innerHTML = entries
    .map(([name, file]) => {
      const percent = clampPercent(file.progress || 0);
      const status = escapeHtml(file.status || "running");
      return `
        <div class="active-video-row ${status}">
          <span class="video-name">${escapeHtml(name)}</span>
          <span class="video-stage">${escapeHtml(file.stage || "Working")}</span>
          <span class="video-bar-track" aria-hidden="true">
            <span class="video-bar" style="width: ${percent}%"></span>
          </span>
          <span class="video-percent">${percent}%</span>
        </div>
      `;
    })
    .join("");
}

function renderQueue(entries) {
  const waiting = entries.filter(([_name, file]) => file.status === "queued").length;
  const completed = entries.filter(([_name, file]) => file.status === "completed").length;
  const canceled = entries.filter(([_name, file]) => file.status === "canceled").length;
  const failed = entries.filter(([_name, file]) => file.status === "failed").length;
  const skipped = entries.filter(([_name, file]) => file.status === "skipped").length;
  elements.queueSummary.textContent = queueSummaryText({
    entries: entries.length,
    completed,
    canceled,
    failed,
    skipped,
    waiting,
  });

  if (entries.length === 0) {
    elements.queueList.innerHTML = `<div class="video-placeholder">${
      state.fileNames.length ? "All listed videos are active." : "Choose a folder to see videos."
    }</div>`;
    return;
  }

  elements.queueList.innerHTML = entries
    .map(([name, file]) => {
      const status = escapeHtml(file.status || "queued");
      const detail = file.detail && file.detail !== statusLabel(file.status) ? file.detail : "";
      return `
        <div class="queue-video-row ${status}">
          <span class="video-status-dot" aria-hidden="true"></span>
          <span class="video-name">${escapeHtml(name)}</span>
          <span class="queue-status">${escapeHtml(statusLabel(file.status))}</span>
          ${detail ? `<span class="queue-detail">${escapeHtml(detail)}</span>` : ""}
        </div>
      `;
    })
    .join("");
}

function setProgress(done, total) {
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;
  setProgressPercent(percent);
}

function setProgressPercent(percent) {
  const value = clampPercent(percent);
  elements.progressBar.style.width = `${value}%`;
  elements.progressPercent.textContent = `${value}%`;
}

function fileFinishedDetail(event, failedStage) {
  if (event.status === "completed") {
    const words = Number(event.word_count || 0).toLocaleString();
    const slides = Number(event.slide_count || 0).toLocaleString();
    return `${words} words, ${slides} slides`;
  }
  if (event.status === "failed") {
    return event.message ? `${failedStage}: ${event.message}` : failedStage;
  }
  if (event.status === "skipped") {
    return event.message || "Skipped";
  }
  return event.message || statusLabel(event.status);
}

function isActiveStatus(status) {
  return status === "preparing" || status === "running";
}

function statusLabel(status) {
  if (status === "completed") return "Complete";
  if (status === "failed") return "Failed";
  if (status === "skipped") return "Skipped";
  if (status === "canceled") return "Canceled";
  if (status === "preparing") return "Preparing";
  if (status === "running") return "Processing";
  if (status === "queued") return state.running ? "Waiting" : "Ready";
  return "Ready";
}

function clampPercent(value) {
  return Math.min(100, Math.max(0, Math.round(Number(value) || 0)));
}

function renderRunControls() {
  elements.cancelRunButton.classList.toggle("hidden", !state.running);
  elements.cancelRunButton.disabled = !state.running || !state.processorStarted || state.cancelRequested;
  elements.cancelRunButton.textContent = state.cancelRequested ? "Canceling..." : "Cancel Run";
}

function updateActiveVideosForCancel() {
  for (const [name, file] of state.files.entries()) {
    if (isActiveStatus(file.status)) {
      state.files.set(name, {
        ...file,
        stage: "Canceling",
        detail: "Stopping",
      });
    }
  }
  renderVideoDashboard();
}

function markRemainingVideosCanceled() {
  let changed = 0;
  for (const [name, file] of dashboardEntries()) {
    if (isActiveStatus(file.status) || file.status === "queued") {
      state.files.set(name, {
        ...file,
        status: "canceled",
        stage: "Canceled",
        detail: "Stopped by user",
        progress: 100,
      });
      changed += 1;
    }
  }
  renderVideoDashboard();
  return changed;
}

function markUnfinishedVideosFailed(detail) {
  let changed = 0;
  for (const [name, file] of dashboardEntries()) {
    if (isActiveStatus(file.status) || file.status === "queued") {
      if (!state.fileOrder.includes(name)) {
        state.fileOrder.push(name);
      }
      state.files.set(name, {
        ...file,
        status: "failed",
        stage: "Not completed",
        detail,
        progress: 100,
      });
      changed += 1;
    }
  }
  renderVideoDashboard();
  return changed;
}

function queueSummaryText(summary) {
  if (summary.entries === 0) return "No videos queued";
  const parts = [];
  if (summary.completed) parts.push(`${summary.completed} complete`);
  if (summary.canceled) parts.push(`${summary.canceled} canceled`);
  if (summary.failed) parts.push(`${summary.failed} failed`);
  if (summary.skipped) parts.push(`${summary.skipped} skipped`);
  if (summary.waiting) parts.push(`${summary.waiting} waiting`);
  return parts.length ? parts.join(" · ") : "All videos active";
}

function loadPersistedSettings() {
  let stored = {};
  try {
    stored = JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY) || "{}");
  } catch {
    stored = {};
  }
  applySettings({ ...DEFAULT_SETTINGS, ...stored });
}

function restoreDefaultSettings() {
  applySettings(DEFAULT_SETTINGS);
  saveCurrentSettings();
  validateSettings({ showDialogOnError: false });
  render();
}

function applySettings(settings) {
  state.recordingSpeed = ["1x", "2x"].includes(settings.recordingSpeed)
    ? settings.recordingSpeed
    : DEFAULT_SETTINGS.recordingSpeed;
  setSelectValue(elements.audioQuality, settings.audioQuality, DEFAULT_SETTINGS.audioQuality);
  setSelectValue(elements.transcriptionEngine, settings.transcriptionEngine, DEFAULT_SETTINGS.transcriptionEngine);
  setSelectValue(elements.whisperModel, settings.whisperModel, DEFAULT_SETTINGS.whisperModel);
  setSelectValue(elements.slideSensitivity, settings.slideSensitivity, DEFAULT_SETTINGS.slideSensitivity);
  elements.concurrentFiles.value = String(validConcurrentFiles(settings.concurrentFiles));
  elements.saveNormalized.checked = Boolean(settings.saveNormalized);
  syncSpeedSegments();
  renderNormalizationWarning();
}

function saveCurrentSettings() {
  const concurrentFiles = Number.parseInt(elements.concurrentFiles.value, 10);
  if (!Number.isInteger(concurrentFiles) || concurrentFiles < 1 || concurrentFiles > 8) return;

  const settings = {
    recordingSpeed: state.recordingSpeed,
    audioQuality: elements.audioQuality.value,
    transcriptionEngine: elements.transcriptionEngine.value,
    whisperModel: elements.whisperModel.value,
    slideSensitivity: elements.slideSensitivity.value,
    concurrentFiles,
    saveNormalized: elements.saveNormalized.checked,
  };

  try {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Settings persistence is helpful, not required for processing.
  }
}

async function cleanupTempFilesAtLaunch() {
  try {
    const outputDir = window.localStorage.getItem(LAST_OUTPUT_STORAGE_KEY) || null;
    await invoke("cleanup_temp_files", { outputDir });
  } catch {
    // Temp cleanup is opportunistic; batch startup performs the same sweep.
  }
}

function rememberOutputDir(path) {
  if (!path) return;
  try {
    window.localStorage.setItem(LAST_OUTPUT_STORAGE_KEY, path);
  } catch {
    // Remembering the last output folder is only used for launch cleanup.
  }
}

function syncSpeedSegments() {
  elements.speedSegments.forEach((button) => {
    button.classList.toggle("active", button.dataset.speed === state.recordingSpeed);
  });
}

function setSelectValue(select, value, fallback) {
  const optionValues = [...select.options].map((option) => option.value);
  select.value = optionValues.includes(value) ? value : fallback;
}

function validConcurrentFiles(value) {
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= 1 && parsed <= 8 ? parsed : DEFAULT_SETTINGS.concurrentFiles;
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

function runDescription() {
  const speed = state.recordingSpeed === "2x" ? "2x to 1x" : "1x";
  const concurrent = Number.parseInt(elements.concurrentFiles.value, 10) || 1;
  return `${state.fileCount} video${state.fileCount === 1 ? "" : "s"} · ${concurrent} at a time · ${speed}`;
}

function startElapsedTimer() {
  stopElapsedTimer();
  state.runStartedAt = Date.now();
  updateElapsedTime();
  state.elapsedTimer = window.setInterval(updateElapsedTime, 1000);
}

function stopElapsedTimer() {
  if (state.elapsedTimer) {
    window.clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }
}

function updateElapsedTime() {
  if (!state.runStartedAt) {
    elements.elapsedTime.textContent = "00:00";
    return;
  }
  const elapsedSeconds = Math.floor((Date.now() - state.runStartedAt) / 1000);
  const minutes = String(Math.floor(elapsedSeconds / 60)).padStart(2, "0");
  const seconds = String(elapsedSeconds % 60).padStart(2, "0");
  elements.elapsedTime.textContent = `${minutes}:${seconds}`;
}

function waitForPaint() {
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => window.requestAnimationFrame(resolve));
  });
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
