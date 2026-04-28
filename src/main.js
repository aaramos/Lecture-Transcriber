const invoke = window.__TAURI__?.core?.invoke;

if (!invoke) {
  throw new Error("Tauri API is not available. Start the app with `cargo tauri dev`.");
}

const STEP_LABELS = {
  Probe: "Checking video",
  Normalize: "Normalizing",
  Audio: "Preparing audio",
  Transcribe: "Transcribing",
  Slides: "Extracting slides",
  Enrich: "Creating study notes",
  Render: "Building HTML",
};

const STEP_SHORT_LABELS = {
  Probe: "Check",
  Normalize: "Normalize",
  Audio: "Audio",
  Transcribe: "Transcript",
  Slides: "Slides",
  Enrich: "Gemini",
  Render: "HTML",
};

const STEP_ORDER = ["Probe", "Normalize", "Audio", "Transcribe", "Slides", "Enrich", "Render"];

const STEP_START_PROGRESS = {
  Probe: 8,
  Normalize: 22,
  Audio: 44,
  Transcribe: 54,
  Slides: 82,
  Enrich: 88,
  Render: 96,
};

const STEP_DONE_PROGRESS = {
  Probe: 18,
  Normalize: 42,
  Audio: 52,
  Transcribe: 78,
  Slides: 96,
  Enrich: 94,
  Render: 100,
};

const SETTINGS_STORAGE_KEY = "lectureProcessor.settings.v2";
const LAST_OUTPUT_STORAGE_KEY = "lectureProcessor.lastOutputDir.v1";
const DEFAULT_SETTINGS = Object.freeze({
  recordingSpeed: "1x",
  audioQuality: "high",
  transcriptionEngine: "faster-whisper",
  transcriptionQuality: "accurate",
  whisperModel: "medium.en",
  slideSensitivity: "medium",
  aiModel: "gemini-2.5-flash",
  enhanceWithGemini: false,
  concurrentFiles: 2,
  saveNormalized: true,
});
const LEGACY_DEFAULT_MIGRATIONS = Object.freeze({
  audioQuality: ["fast", DEFAULT_SETTINGS.audioQuality],
  transcriptionQuality: ["balanced", DEFAULT_SETTINGS.transcriptionQuality],
  whisperModel: ["large-v3", DEFAULT_SETTINGS.whisperModel],
  aiModel: ["gemini-2.5-flash-lite", DEFAULT_SETTINGS.aiModel],
});

const state = {
  folderMode: "source",
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
  systemMetricsTimer: null,
  finishedFileReports: 0,
  geminiKeySaved: false,
  enhanceWithGeminiPreference: DEFAULT_SETTINGS.enhanceWithGemini,
  dependencyReady: false,
  dependencySetupRunning: false,
  files: new Map(),
  fileOrder: [],
  skippedFiles: new Set(),
  autoSkipReasons: new Map(),
};

const elements = {
  setupScreen: document.querySelector("#setupScreen"),
  setupStatusTitle: document.querySelector("#setupStatusTitle"),
  setupStatusMeta: document.querySelector("#setupStatusMeta"),
  setupProgressBar: document.querySelector("#setupProgressBar"),
  setupProgressPercent: document.querySelector("#setupProgressPercent"),
  setupStepList: document.querySelector("#setupStepList"),
  setupInstallButton: document.querySelector("#setupInstallButton"),
  setupRecheckButton: document.querySelector("#setupRecheckButton"),
  setupError: document.querySelector("#setupError"),
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
  cpuMetric: document.querySelector("#cpuMetric"),
  cpuMetricStatus: document.querySelector("#cpuMetricStatus"),
  gpuMetric: document.querySelector("#gpuMetric"),
  gpuMetricStatus: document.querySelector("#gpuMetricStatus"),
  memoryMetric: document.querySelector("#memoryMetric"),
  memoryMetricStatus: document.querySelector("#memoryMetricStatus"),
  activeVideos: document.querySelector("#activeVideos"),
  activeSummary: document.querySelector("#activeSummary"),
  completedCount: document.querySelector("#completedCount"),
  processingCount: document.querySelector("#processingCount"),
  waitingCount: document.querySelector("#waitingCount"),
  failedCount: document.querySelector("#failedCount"),
  skippedCount: document.querySelector("#skippedCount"),
  canceledCount: document.querySelector("#canceledCount"),
  audioQuality: document.querySelector("#audioQuality"),
  transcriptionEngine: document.querySelector("#transcriptionEngine"),
  transcriptionQuality: document.querySelector("#transcriptionQuality"),
  whisperModel: document.querySelector("#whisperModel"),
  slideSensitivity: document.querySelector("#slideSensitivity"),
  enhanceWithGemini: document.querySelector("#enhanceWithGemini"),
  aiModel: document.querySelector("#aiModel"),
  geminiApiKey: document.querySelector("#geminiApiKey"),
  saveGeminiKeyButton: document.querySelector("#saveGeminiKeyButton"),
  geminiKeyStatus: document.querySelector("#geminiKeyStatus"),
  concurrentFiles: document.querySelector("#concurrentFiles"),
  saveNormalized: document.querySelector("#saveNormalized"),
  restoreDefaultsButton: document.querySelector("#restoreDefaultsButton"),
  speedSegments: [...document.querySelectorAll(".segment")],
};

window.__TAURI__?.event?.listen?.("processor-event", (event) => handleProcessorEvent(event.payload));
window.__TAURI__?.event?.listen?.("dependency-event", (event) => handleDependencyEvent(event.payload));
loadPersistedSettings();
refreshDependencyStatus();
refreshGeminiKeyStatus();
cleanupTempFilesAtLaunch();
setupDragAndDrop();
startSystemMetrics();

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
elements.setupInstallButton.addEventListener("click", setupDependencies);
elements.setupRecheckButton.addEventListener("click", refreshDependencyStatus);
elements.cancelRunButton.addEventListener("click", cancelBatch);
elements.openOutputButton.addEventListener("click", openOutput);
elements.settingsButton.addEventListener("click", () => showDialog(elements.settingsDialog));
elements.restoreDefaultsButton.addEventListener("click", restoreDefaultSettings);
elements.saveGeminiKeyButton.addEventListener("click", () => saveGeminiKey());
elements.activeVideos.addEventListener("click", (event) => {
  const button = event.target.closest("[data-video-action]");
  if (!button) return;
  handleVideoAction(button.dataset.videoAction, button.dataset.source);
});
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
  elements.transcriptionQuality,
  elements.whisperModel,
  elements.slideSensitivity,
  elements.enhanceWithGemini,
  elements.aiModel,
  elements.saveNormalized,
].forEach((element) => {
  element.addEventListener("change", () => {
    if (element === elements.enhanceWithGemini && state.folderMode !== "processed") {
      state.enhanceWithGeminiPreference = elements.enhanceWithGemini.checked;
    }
    saveCurrentSettings();
    renderAiControls();
    render();
  });
});

elements.speedSegments.forEach((button) => {
  button.addEventListener("click", () => {
    state.recordingSpeed = button.dataset.speed;
    elements.speedSegments.forEach((item) => item.classList.toggle("active", item === button));
    saveCurrentSettings();
    render();
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
  if (state.inputDir) {
    await scanFolder();
  }
  render();
}

async function scanFolder() {
  try {
    const scan = await invoke("scan_folder", { inputDir: state.inputDir, outputDir: state.outputDir || null });
    state.folderMode = scan.folderMode === "processed" ? "processed" : "source";
    state.outputDir = scan.outputDir || state.outputDir;
    rememberOutputDir(state.outputDir);
    state.fileNames = state.folderMode === "processed" ? scan.lectureFiles || [] : scan.movFiles || [];
    state.fileCount = state.fileNames.length;
    state.skippedFiles.clear();
    state.autoSkipReasons.clear();
    (scan.alreadyProcessedFiles || []).forEach((name) => {
      state.skippedFiles.add(name);
      state.autoSkipReasons.set(name, "Already processed");
    });
    (scan.alreadyEnhancedFiles || []).forEach((name) => {
      state.skippedFiles.add(name);
      state.autoSkipReasons.set(name, "Already enhanced");
    });
    initializeQueuedFiles();
    const emptyMessage =
      state.folderMode === "processed"
        ? "No completed processed lectures found. Try a different folder."
        : "No .mov files found. Try a different folder.";
    setFolderError(state.fileCount === 0 ? emptyMessage : "");
  } catch (error) {
    state.folderMode = "source";
    state.fileCount = 0;
    state.fileNames = [];
    state.files.clear();
    state.fileOrder = [];
    state.skippedFiles.clear();
    state.autoSkipReasons.clear();
    setFolderError(String(error));
  }
}

function clearFolder() {
  state.folderMode = "source";
  state.inputDir = "";
  state.outputDir = "";
  state.fileCount = 0;
  state.fileNames = [];
  state.files.clear();
  state.fileOrder = [];
  state.skippedFiles.clear();
  state.autoSkipReasons.clear();
  setFolderError("");
  render();
}

async function startBatch() {
  if (state.running || !state.inputDir || processableFileCount() === 0) return;
  if (!state.dependencyReady) {
    await refreshDependencyStatus();
    if (!state.dependencyReady) return;
  }

  if (requiresNormalizationConfirmation()) {
    const confirmed = await confirmNormalization();
    if (!confirmed) return;
  }

  if (needsGemini() && elements.geminiApiKey.value.trim()) {
    const saved = await saveGeminiKey({ quiet: true });
    if (!saved) return;
  }
  await refreshGeminiKeyStatus();
  const finalConcurrentFiles = validateSettings();
  if (!finalConcurrentFiles) return;

  resetResults();
  setRunning(true);
  primeQueuedFiles(finalConcurrentFiles);
  renderVideoDashboard();
  await waitForPaint();

  const request = {
    mode: state.folderMode === "processed" ? "enhance" : "process",
    inputDir: state.inputDir,
    outputDir: state.outputDir,
    recordingSpeed: state.recordingSpeed,
    confirmNormalization: requiresNormalizationConfirmation(),
    concurrentFiles: finalConcurrentFiles,
    saveNormalizedVideo: elements.saveNormalized.checked,
    audioQuality: elements.audioQuality.value,
    transcriptionEngine: elements.transcriptionEngine.value,
    transcriptionQuality: elements.transcriptionQuality.value,
    whisperModel: elements.whisperModel.value,
    slideSensitivity: elements.slideSensitivity.value,
    aiProvider: needsGemini() ? "gemini" : "none",
    aiModel: elements.aiModel.value,
    minDuration: 60,
    skippedFiles: manualSkippedFiles(),
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

async function refreshDependencyStatus() {
  try {
    const status = await invoke("dependency_status");
    applyDependencyStatus(status);
  } catch (error) {
    showDependencySetup({
      message: "Dependency check failed",
      detail: String(error),
      progress: 0,
      failed: true,
    });
  }
}

async function setupDependencies() {
  if (state.dependencySetupRunning) return;
  state.dependencySetupRunning = true;
  elements.setupInstallButton.disabled = true;
  elements.setupRecheckButton.disabled = true;
  setSetupError("");
  updateSetupProgress({
    title: "Starting setup",
    detail: "Preparing the app support folder.",
    progress: 2,
    step: 0,
  });

  try {
    const status = await invoke("setup_dependencies");
    applyDependencyStatus(status);
  } catch (error) {
    showDependencySetup({
      message: "Setup failed",
      detail: String(error),
      progress: 100,
      failed: true,
    });
  } finally {
    state.dependencySetupRunning = false;
    elements.setupInstallButton.disabled = false;
    elements.setupRecheckButton.disabled = false;
  }
}

function applyDependencyStatus(status) {
  state.dependencyReady = Boolean(status?.ready);
  if (state.dependencyReady) {
    elements.setupScreen.classList.add("hidden");
    render();
    return;
  }

  showDependencySetup({
    message: status?.message || "Dependencies need setup",
    detail: status?.detail || "Install the local processor runtime before starting a batch.",
    progress: 0,
    failed: false,
  });
}

function showDependencySetup({ message, detail, progress, failed }) {
  state.dependencyReady = false;
  elements.setupScreen.classList.remove("hidden");
  elements.setupStatusTitle.textContent = message;
  elements.setupStatusMeta.textContent = detail;
  setSetupProgress(progress || 0);
  setSetupError(failed ? detail : "");
  markSetupSteps(failed ? 3 : 0, failed);
}

function handleDependencyEvent(event) {
  if (!event || typeof event !== "object") return;
  updateSetupProgress({
    title: event.title || "Installing dependencies",
    detail: event.detail || "",
    progress: event.progress || 0,
    step: event.step || 0,
    failed: event.kind === "failed",
  });
  if (event.kind === "failed") {
    setSetupError(event.detail || "Setup failed.");
  }
}

function updateSetupProgress({ title, detail, progress, step, failed }) {
  elements.setupScreen.classList.remove("hidden");
  elements.setupStatusTitle.textContent = title;
  elements.setupStatusMeta.textContent = detail;
  setSetupProgress(progress);
  markSetupSteps(step, failed);
}

function setSetupProgress(progress) {
  const value = clampPercent(progress);
  elements.setupProgressBar.style.width = `${value}%`;
  elements.setupProgressPercent.textContent = `${value}%`;
}

function markSetupSteps(activeIndex, failed) {
  [...elements.setupStepList.children].forEach((step, index) => {
    step.classList.toggle("done", !failed && index < activeIndex);
    step.classList.toggle("active", !failed && index === activeIndex);
    step.classList.toggle("failed", Boolean(failed && index === activeIndex));
  });
}

function setSetupError(message) {
  elements.setupError.textContent = message;
  elements.setupError.classList.toggle("hidden", !message);
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
    await invoke("open_batch_output", { outputDir: path });
  } catch (error) {
    elements.runMeta.textContent = `Could not open output: ${error}`;
  }
}

function applyProcessorResult(result) {
  state.lastOutputDir = result.outputDir || state.outputDir;
  rememberOutputDir(state.lastOutputDir);

  let unfinished = 0;
  if (result.cancelled) {
    unfinished = markRemainingVideosStopped();
  } else {
    unfinished = markUnfinishedVideosFailed("Processor ended before reporting this video complete.");
  }

  renderVideoDashboard();
  const counts = countDashboardEntries(dashboardEntries());
  const exitCode = Number(result.exitCode || 0);
  const hasFailures = counts.failed > 0 || exitCode !== 0;
  const hasSkips = counts.skipped > 0;
  const hasStopped = counts.stopped > 0;
  const failedBeforeFileResults = exitCode !== 0 && state.finishedFileReports === 0;
  const allReported =
    counts.processing === 0 &&
    counts.waiting === 0 &&
    counts.stopped === 0 &&
    counts.failed === 0 &&
    counts.completed + counts.skipped === (state.progressTotal || state.fileCount || counts.total);

  if (result.cancelled) {
    elements.runTitle.textContent = "Batch stopped";
    elements.statusPill.textContent = "Stopped";
    elements.statusPill.className = "status-pill warning";
    elements.runMeta.textContent = "Processing was stopped. Completed files remain in the output folder.";
  } else if (exitCode === 0 && allReported && !hasSkips && !hasStopped) {
    elements.runTitle.textContent = "Batch finished";
    elements.statusPill.textContent = "Complete";
    elements.statusPill.className = "status-pill complete";
    elements.runMeta.textContent = "All videos finished. Open the output folder for transcripts, videos, and slides.";
  } else {
    elements.runTitle.textContent = failedBeforeFileResults
        ? "Batch failed"
        : hasFailures
          ? "Batch finished with issues"
          : hasStopped
            ? "Batch finished with stopped files"
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
  if (!result.cancelled && exitCode === 0 && state.lastOutputDir) {
    openOutput();
  }
}

function finalRunSummary(counts, unfinished, exitCode) {
  const parts = [];
  if (counts.completed) parts.push(`${counts.completed} complete`);
  if (counts.failed) parts.push(`${counts.failed} failed`);
  if (counts.skipped) parts.push(`${counts.skipped} skipped`);
  if (counts.stopped) parts.push(`${counts.stopped} stopped`);
  if (unfinished) parts.push(`${unfinished} did not report a final result`);
  if (exitCode !== 0 && counts.failed === 0) parts.push(`processor exited with code ${exitCode}`);
  return `${parts.join(" · ") || "Run needs review"}. Open the output folder for batch_summary.txt or batch_error.txt.`;
}

function render() {
  const hasFolder = Boolean(state.inputDir);
  const noun = itemNoun();
  elements.folderTitle.textContent = hasFolder ? basename(state.inputDir) : "Choose lecture or processed folder";
  elements.folderSub.textContent = hasFolder
    ? `${state.inputDir} · ${state.fileCount} ${noun}${state.fileCount === 1 ? "" : "s"}`
    : "No folder selected";
  elements.outputPath.textContent = state.outputDir || "-";
  elements.clearFolderButton.classList.toggle("hidden", !hasFolder);
  elements.startButton.disabled = state.running || !hasFolder || processableFileCount() === 0;
  elements.startButton.textContent = state.folderMode === "processed" ? "Enhance" : "Start";
  elements.chooseOutputButton.disabled = state.running || state.folderMode === "processed";
  elements.enhanceWithGemini.checked =
    state.folderMode === "processed" ? true : state.enhanceWithGeminiPreference;
  elements.enhanceWithGemini.disabled = state.running || state.folderMode === "processed";
  if (!state.running) {
    syncPendingFilePlans();
    elements.runMeta.textContent = hasFolder
      ? readyRunMeta()
      : "Select a folder to begin.";
  }
  renderVideoDashboard();
  renderNormalizationWarning();
}

function setRunning(running) {
  state.running = running;
  elements.startButton.disabled = running || !state.inputDir || processableFileCount() === 0;
  elements.chooseFolderButton.disabled = running;
  elements.clearFolderButton.disabled = running;
  elements.chooseOutputButton.disabled = running || state.folderMode === "processed";
  elements.enhanceWithGemini.disabled = running || state.folderMode === "processed";
  elements.progressBar.classList.toggle("running", running);
  renderRunControls();
  if (running) {
    const ready = processableFileCount();
    const noun = itemNoun();
    elements.runTitle.textContent = `Preparing ${ready} ${noun}${ready === 1 ? "" : "s"}...`;
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
  elements.waitingCount.textContent = String(processableFileCount() || 0);
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

function plannedStepNames() {
  if (state.folderMode === "processed") {
    return ["Enrich", "Render"];
  }

  const steps = ["Probe"];
  if (state.recordingSpeed === "2x") {
    steps.push("Normalize");
  }
  if (elements.transcriptionEngine.value !== "none") {
    steps.push("Audio");
  }
  steps.push("Transcribe", "Slides");
  if (needsGemini()) {
    steps.push("Enrich");
  }
  steps.push("Render");
  return steps;
}

function buildInitialSteps(fileStatus = "queued") {
  const stepStatus = fileStatus === "skipped" ? "skipped" : "waiting";
  return plannedStepNames().map((key) => ({
    key,
    label: STEP_SHORT_LABELS[key] || key,
    status: stepStatus,
    elapsedSeconds: null,
    startedAt: null,
  }));
}

function stepRunningDetail(step) {
  const label = STEP_LABELS[step] || step || "Working";
  return `${label} started`;
}

function markStepStarted(file, step) {
  const now = Date.now();
  const steps = ensureStepStates(file, step);
  const activeIndex = steps.findIndex((item) => item.key === step);
  return {
    steps: steps.map((item, index) => {
      if (item.key === step) {
        return {
          ...item,
          status: "active",
          startedAt: now,
          elapsedSeconds: null,
        };
      }
      if (index < activeIndex && ["waiting", "active"].includes(item.status)) {
        return {
          ...item,
          status: "done",
          elapsedSeconds: item.elapsedSeconds,
        };
      }
      return item;
    }),
    currentStep: step,
    currentStepStartedAt: now,
    lastEventAt: now,
  };
}

function markStepFinished(file, step, elapsedSeconds) {
  const now = Date.now();
  const steps = ensureStepStates(file, step);
  return {
    steps: steps.map((item) =>
      item.key === step
        ? {
            ...item,
            status: "done",
            elapsedSeconds,
            startedAt: item.startedAt || null,
          }
        : item,
    ),
    currentStep: file?.currentStep === step ? null : file?.currentStep || null,
    currentStepStartedAt: file?.currentStep === step ? null : file?.currentStepStartedAt || null,
    lastEventAt: now,
  };
}

function markFileFinished(file, status, failureStep) {
  const steps = ensureStepStates(file, failureStep || file?.currentStep || null);
  if (status === "completed") {
    return {
      steps: steps.map((item) => ({
        ...item,
        status: item.status === "skipped" ? "skipped" : "done",
      })),
      currentStep: null,
      currentStepStartedAt: null,
      lastEventAt: Date.now(),
    };
  }

  if (status === "failed") {
    const failedKey = failureStep || file?.currentStep || steps.find((item) => item.status === "active")?.key;
    return {
      steps: steps.map((item) =>
        item.key === failedKey
          ? { ...item, status: "failed" }
          : item.status === "active"
            ? { ...item, status: "failed" }
            : item,
      ),
      currentStep: null,
      currentStepStartedAt: null,
      lastEventAt: Date.now(),
    };
  }

  if (status === "skipped" || status === "stopped") {
    return {
      steps: steps.map((item) =>
        item.status === "done"
          ? item
          : {
              ...item,
              status: status === "stopped" ? "stopped" : "skipped",
            },
      ),
      currentStep: null,
      currentStepStartedAt: null,
      lastEventAt: Date.now(),
    };
  }

  return {};
}

function ensureStepStates(file, step = null) {
  const existing = Array.isArray(file?.steps) && file.steps.length > 0 ? file.steps : buildInitialSteps(file?.status);
  if (!step || existing.some((item) => item.key === step)) {
    return existing;
  }

  const inserted = [
    ...existing,
    {
      key: step,
      label: STEP_SHORT_LABELS[step] || step,
      status: "waiting",
      elapsedSeconds: null,
      startedAt: null,
    },
  ];
  return inserted.sort((a, b) => stepOrderIndex(a.key) - stepOrderIndex(b.key));
}

function stepOrderIndex(step) {
  const index = STEP_ORDER.indexOf(step);
  return index === -1 ? STEP_ORDER.length : index;
}

function renderStepStrip(file) {
  const steps = ensureStepStates(file);
  if (!steps.length) return "";
  const html = steps
    .map((step) => {
      const stateClass = escapeHtml(step.status || "waiting");
      const activeTime = step.status === "active" ? activeStepDuration(file, step) : "";
      const finishedTime = step.status === "done" && step.elapsedSeconds ? formatDuration(step.elapsedSeconds) : "";
      const time = activeTime || finishedTime;
      return `<span class="video-step ${stateClass}"><span>${escapeHtml(step.label)}</span>${
        time ? `<span class="video-step-time">${escapeHtml(time)}</span>` : ""
      }</span>`;
    })
    .join("");
  return `<div class="video-step-list" aria-label="File steps">${html}</div>`;
}

function activeStepDetail(file) {
  const activeStep = ensureStepStates(file).find((step) => step.status === "active");
  if (!activeStep) return file.detail || statusLabel(file.status);
  return `${STEP_LABELS[activeStep.key] || activeStep.label} for ${activeStepDuration(file, activeStep)}`;
}

function activeStepDuration(file, step) {
  const startedAt = step.startedAt || file?.currentStepStartedAt;
  if (!startedAt) return "0s";
  return formatDuration((Date.now() - startedAt) / 1000);
}

function formatDuration(seconds) {
  const wholeSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
  if (wholeSeconds < 60) return `${wholeSeconds}s`;
  const minutes = Math.floor(wholeSeconds / 60);
  const remainingSeconds = wholeSeconds % 60;
  if (minutes < 60) return `${minutes}m ${String(remainingSeconds).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${String(remainingMinutes).padStart(2, "0")}m`;
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
  const valid = Number.isInteger(value) && value >= 1 && value <= 3;
  elements.concurrentFiles.classList.toggle("invalid", !valid);
  if (!valid) {
    if (showDialogOnError) {
      showDialog(elements.settingsDialog);
    }
    elements.concurrentFiles.focus();
    elements.runMeta.textContent = "Concurrent files must be between 1 and 3.";
    return null;
  }
  if (needsGemini() && !state.geminiKeySaved && !elements.geminiApiKey.value.trim()) {
    if (showDialogOnError) {
      showDialog(elements.settingsDialog);
    }
    elements.runMeta.textContent = "Gemini needs an API key saved in Keychain.";
    elements.geminiApiKey.focus();
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
    elements.runTitle.textContent = `${state.folderMode === "processed" ? "Enhancing" : "Processing"} ${basename(state.inputDir)}`;
    elements.runMeta.textContent = runDescription();
    renderRunControls();
    renderVideoDashboard();
  } else if (event.kind === "file_started") {
    if (["skipped", "stopping", "stopped"].includes(state.files.get(event.source)?.status)) {
      return;
    }
    updateFile(event.source, {
      status: "running",
      stage: "Preparing",
      detail: "Preparing video",
      progress: 5,
    });
  } else if (event.kind === "step_started") {
    if (["skipped", "stopping", "stopped"].includes(state.files.get(event.source)?.status)) {
      return;
    }
    updateFile(event.source, {
      status: "running",
      stage: STEP_LABELS[event.step] || event.step,
      detail: stepRunningDetail(event.step),
      progress: STEP_START_PROGRESS[event.step] || 12,
      ...markStepStarted(state.files.get(event.source), event.step),
    });
  } else if (event.kind === "step_finished") {
    if (["skipped", "stopping", "stopped"].includes(state.files.get(event.source)?.status)) {
      return;
    }
    updateFile(event.source, {
      stage: `${STEP_LABELS[event.step] || event.step} complete`,
      detail: `${Number(event.elapsed_seconds || 0).toFixed(1)}s`,
      progress: STEP_DONE_PROGRESS[event.step] || 20,
      ...markStepFinished(state.files.get(event.source), event.step, Number(event.elapsed_seconds || 0)),
    });
  } else if (event.kind === "enrichment_started") {
    updateFile(event.source, {
      detail: "Waiting for Gemini",
      ...markStepStarted(state.files.get(event.source), "Enrich"),
    });
  } else if (event.kind === "enrichment_progress") {
    const completed = Number(event.completed || 0);
    const total = Number(event.total || 0);
    const countText = total > 0 ? ` (${completed}/${total})` : "";
    const current = state.files.get(event.source);
    const activePatch = current?.currentStep === "Enrich" ? {} : markStepStarted(current, "Enrich");
    updateFile(event.source, {
      status: "running",
      stage: STEP_LABELS.Enrich,
      detail: `Gemini: ${event.step || "working"}${countText}`,
      progress: Math.max(STEP_START_PROGRESS.Enrich, Math.min(STEP_DONE_PROGRESS.Enrich, 88 + completed)),
      ...activePatch,
    });
  } else if (event.kind === "enrichment_finished") {
    updateFile(event.source, {
      detail: event.title ? `Gemini finished: ${event.title}` : "Gemini finished",
      ...markStepFinished(state.files.get(event.source), "Enrich", Number(event.elapsed_seconds || 0)),
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
      ...markFileFinished(previous, event.status, event.failure_step),
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
    stage: "Queue",
    detail: "Waiting to start",
    progress: 0,
    steps: buildInitialSteps("queued"),
  };
  state.files.set(source, { ...previous, ...patch });
  renderVideoDashboard();
}

function renderVideoDashboard() {
  const entries = dashboardEntries();
  renderCounts(entries);
  renderOverallProgress(entries);
  renderVideoList(entries);
}

function primeQueuedFiles(concurrentFiles) {
  state.files.clear();
  const names = state.fileNames.length
    ? state.fileNames
    : Array.from({ length: state.fileCount }, (_value, index) => `File ${index + 1}`);

  state.fileOrder = names;
  let activeSlots = 0;
  names.forEach((name, index) => {
    const skipped = state.skippedFiles.has(name);
    const preparing = !skipped && activeSlots < concurrentFiles;
    const skipReason = state.autoSkipReasons.get(name) || "Skipped by user";
    if (preparing) activeSlots += 1;
    state.files.set(name, {
      status: skipped ? "skipped" : preparing ? "preparing" : "queued",
      stage: skipped ? "Skipped" : preparing ? "Preparing" : "Queue",
      detail: skipped ? skipReason : preparing ? `Preparing ${itemNoun()}` : "Waiting to start",
      progress: skipped ? 100 : preparing ? 3 : 0,
      steps: buildInitialSteps(skipped ? "skipped" : "queued"),
    });
  });

  state.progressTotal = names.length;
  const activeLimit = Math.min(concurrentFiles, processableFileCount());
  const noun = itemNoun();
  elements.runMeta.textContent = `${state.folderMode === "processed" ? "Enhancing" : "Processing"} up to ${activeLimit} ${noun}${
    activeLimit === 1 ? "" : "s"
  } at once.`;
  renderVideoDashboard();
}

function initializeQueuedFiles() {
  state.files.clear();
  state.fileOrder = state.fileNames.length ? [...state.fileNames] : [];
  state.fileOrder.forEach((name) => {
    const skipped = state.skippedFiles.has(name);
    const skipReason = state.autoSkipReasons.get(name) || "Skipped by user";
    state.files.set(name, {
      status: skipped ? "skipped" : "queued",
      stage: skipped ? "Skipped" : "Queue",
      detail: skipped ? skipReason : "Waiting to start",
      progress: skipped ? 100 : 0,
      steps: buildInitialSteps(skipped ? "skipped" : "queued"),
    });
  });
}

function syncPendingFilePlans() {
  if (state.finishedFileReports > 0) return;
  for (const [name, file] of state.files.entries()) {
    if (!["queued", "skipped", "preparing"].includes(file.status)) continue;
    state.files.set(name, {
      ...file,
      steps: buildInitialSteps(file.status === "skipped" ? "skipped" : "queued"),
      currentStep: null,
      currentStepStartedAt: null,
    });
  }
}

function dashboardEntries() {
  if (state.fileOrder.length > 0) {
    return state.fileOrder.map((name) => [
      name,
      state.files.get(name) || {
        status: "queued",
        stage: "Queue",
        detail: "Waiting to start",
        progress: 0,
        steps: buildInitialSteps("queued"),
      },
    ]);
  }

  if (state.fileNames.length > 0) {
    return state.fileNames.map((name) => [
      name,
      {
        status: "queued",
        stage: "Queue",
        detail: "Waiting to start",
        progress: 0,
        steps: buildInitialSteps("queued"),
      },
    ]);
  }

  return [];
}

function renderCounts(entries) {
  const { completed, processing, failed, skipped, stopped, waiting } = countDashboardEntries(entries);

  elements.completedCount.textContent = String(completed);
  elements.processingCount.textContent = String(processing);
  elements.waitingCount.textContent = String(waiting);
  elements.failedCount.textContent = String(failed);
  elements.skippedCount.textContent = String(skipped);
  elements.canceledCount.textContent = String(stopped);
}

function countDashboardEntries(entries) {
  return {
    total: entries.length,
    completed: entries.filter(([_name, file]) => file.status === "completed").length,
    processing: entries.filter(([_name, file]) => isActiveStatus(file.status)).length,
    failed: entries.filter(([_name, file]) => file.status === "failed").length,
    skipped: entries.filter(([_name, file]) => file.status === "skipped").length,
    stopped: entries.filter(([_name, file]) => file.status === "stopped").length,
    waiting: entries.filter(([_name, file]) => file.status === "queued").length,
  };
}

function renderOverallProgress(entries) {
  const total = state.progressTotal || entries.length || state.fileCount || 0;
  const knownProgress = entries.reduce((sum, [_name, file]) => sum + Number(file.progress || 0), 0);
  const percent = total > 0 ? Math.round(knownProgress / total) : 0;
  setProgressPercent(percent);
}

function renderVideoList(entries) {
  elements.activeSummary.textContent = videoListSummaryText(countDashboardEntries(entries));
  if (entries.length === 0) {
    elements.activeVideos.innerHTML = `<div class="video-placeholder">Choose a folder to see files.</div>`;
    return;
  }

  elements.activeVideos.innerHTML = entries
    .map(([name, file]) => {
      const status = escapeHtml(file.status || "queued");
      const percent = clampPercent(file.progress || 0);
      const active = isActiveStatus(file.status);
      const detail = fileFinishedDetailFromState(file);
      const action = videoActionFor(name, file);
      return `
        <div class="video-row ${status}">
          <span class="video-status-dot" aria-hidden="true"></span>
          <div class="video-main">
            <div class="video-title-line">
              <span class="video-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
              <span class="video-stage">${escapeHtml(file.stage || statusLabel(file.status))}</span>
            </div>
            <div class="video-detail">${escapeHtml(detail)}</div>
            ${renderStepStrip(file)}
          </div>
          ${
            active
              ? `<span class="video-progress-cell"><span class="video-bar-track" aria-hidden="true"><span class="video-bar" style="width: ${percent}%"></span></span><span class="video-percent">${percent}%</span></span>`
              : `<span class="video-progress-cell complete"><span class="video-percent">${percent}%</span></span>`
          }
          ${
            action
              ? `<button class="${action.className}" type="button" data-video-action="${action.action}" data-source="${escapeHtml(name)}">${action.label}</button>`
              : `<span class="video-row-spacer" aria-hidden="true"></span>`
          }
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
  if (event.status === "stopped") {
    return event.message || "Stopped by user";
  }
  return event.message || statusLabel(event.status);
}

function fileFinishedDetailFromState(file) {
  if (!file) return "";
  if (file.status === "completed") return file.detail || "Complete";
  if (file.status === "failed") return file.detail || "Needs review";
  if (file.status === "skipped") return file.detail || "Skipped by user";
  if (file.status === "stopped") return file.detail || "Stopped by user";
  if (file.status === "queued") return file.detail || "Waiting to start";
  if (isActiveStatus(file.status)) return activeStepDetail(file);
  return file.detail || statusLabel(file.status);
}

function isActiveStatus(status) {
  return status === "preparing" || status === "running" || status === "stopping";
}

function statusLabel(status) {
  if (status === "completed") return "Complete";
  if (status === "failed") return "Failed";
  if (status === "skipped") return "Skipped";
  if (status === "stopped") return "Stopped";
  if (status === "stopping") return "Stopping";
  if (status === "preparing") return "Preparing";
  if (status === "running") return "Processing";
  if (status === "queued") return "Queue";
  return "Queue";
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
        status: "stopping",
        stage: "Stopping",
        detail: "Stopping batch",
        ...markFileFinished(file, "stopped"),
      });
    }
  }
  renderVideoDashboard();
}

function markRemainingVideosStopped() {
  let changed = 0;
  for (const [name, file] of dashboardEntries()) {
    if (isActiveStatus(file.status) || file.status === "queued") {
      state.files.set(name, {
        ...file,
        status: "stopped",
        stage: "Stopped",
        detail: "Stopped by user",
        progress: 100,
        ...markFileFinished(file, "stopped"),
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
        ...markFileFinished(file, "failed"),
      });
      changed += 1;
    }
  }
  renderVideoDashboard();
  return changed;
}

function videoListSummaryText(summary) {
  if (summary.total === 0) return "Choose a folder to see files";
  const parts = [];
  if (summary.completed) parts.push(`${summary.completed} complete`);
  if (summary.processing) parts.push(`${summary.processing} active`);
  if (summary.waiting) parts.push(`${summary.waiting} queued`);
  if (summary.failed) parts.push(`${summary.failed} failed`);
  if (summary.skipped) parts.push(`${summary.skipped} skipped`);
  if (summary.stopped) parts.push(`${summary.stopped} stopped`);
  return parts.length ? parts.join(" · ") : "No videos ready";
}

function videoActionFor(name, file) {
  if (!name || !file) return null;
  if (!state.running) {
    if (file.status === "skipped") {
      if (state.autoSkipReasons.has(name)) return null;
      return { action: "unskip", label: "Use", className: "video-action-button" };
    }
    if (file.status === "queued") {
      return { action: "skip", label: "Skip", className: "video-action-button" };
    }
    return null;
  }

  if (file.status === "queued") {
    return { action: "skip", label: "Skip", className: "video-action-button" };
  }
  if (file.status === "preparing" || file.status === "running") {
    return { action: "stop", label: "Stop", className: "video-action-button danger" };
  }
  return null;
}

async function handleVideoAction(action, source) {
  if (!source || !action) return;
  if (!state.running) {
    if (action === "skip") {
      state.skippedFiles.add(source);
      updateFile(source, {
        status: "skipped",
        stage: "Skipped",
        detail: "Skipped by user",
        progress: 100,
        ...markFileFinished(state.files.get(source), "skipped"),
      });
      render();
    } else if (action === "unskip") {
      state.skippedFiles.delete(source);
      updateFile(source, {
        status: "queued",
        stage: "Queue",
        detail: "Waiting to start",
        progress: 0,
        steps: buildInitialSteps("queued"),
        currentStep: null,
        currentStepStartedAt: null,
      });
      render();
    }
    return;
  }

  if (action === "skip") {
    updateFile(source, {
      status: "skipped",
      stage: "Skipped",
      detail: "Skipped by user",
      progress: 100,
      ...markFileFinished(state.files.get(source), "skipped"),
    });
    await updateFileControl("skip", source);
  } else if (action === "stop") {
    updateFile(source, {
      status: "stopping",
      stage: "Stopping",
      detail: "Stopping and removing partial output",
      ...markFileFinished(state.files.get(source), "stopped"),
    });
    await updateFileControl("stop", source);
  }
}

async function updateFileControl(action, source) {
  try {
    await invoke("update_file_control", {
      request: {
        outputDir: state.outputDir,
        source,
        action,
      },
    });
  } catch (error) {
    updateFile(source, {
      status: "failed",
      stage: "Control failed",
      detail: `Could not ${action} file: ${error}`,
      progress: 100,
    });
  }
}

function processableFileCount() {
  const names = state.fileNames.length ? state.fileNames : state.fileOrder;
  return names.filter((name) => !state.skippedFiles.has(name)).length;
}

function manualSkippedFiles() {
  return [...state.skippedFiles].filter((name) => !state.autoSkipReasons.has(name));
}

function needsGemini() {
  return state.folderMode === "processed" || elements.enhanceWithGemini.checked;
}

function requiresNormalizationConfirmation() {
  return state.folderMode !== "processed" && state.recordingSpeed === "2x";
}

function itemNoun() {
  return state.folderMode === "processed" ? "lecture" : "video";
}

function readyRunMeta() {
  const ready = processableFileCount();
  const skipped = state.skippedFiles.size;
  const noun = itemNoun();
  const readyText = `${ready} ${noun}${ready === 1 ? "" : "s"} ready`;
  return skipped ? `${readyText} · ${skipped} skipped` : readyText;
}

function loadPersistedSettings() {
  let stored = {};
  try {
    stored = JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY) || "{}");
  } catch {
    stored = {};
  }
  applySettings(migratePersistedSettings({ ...DEFAULT_SETTINGS, ...stored }));
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
  setSelectValue(elements.transcriptionQuality, settings.transcriptionQuality, DEFAULT_SETTINGS.transcriptionQuality);
  setSelectValue(elements.whisperModel, settings.whisperModel, DEFAULT_SETTINGS.whisperModel);
  setSelectValue(elements.slideSensitivity, settings.slideSensitivity, DEFAULT_SETTINGS.slideSensitivity);
  setSelectValue(elements.aiModel, settings.aiModel, DEFAULT_SETTINGS.aiModel);
  state.enhanceWithGeminiPreference = Boolean(settings.enhanceWithGemini);
  elements.enhanceWithGemini.checked = state.enhanceWithGeminiPreference;
  elements.concurrentFiles.value = String(validConcurrentFiles(settings.concurrentFiles));
  elements.saveNormalized.checked = Boolean(settings.saveNormalized);
  syncSpeedSegments();
  renderAiControls();
  renderNormalizationWarning();
}

function migratePersistedSettings(settings) {
  const migrated = { ...settings };
  Object.entries(LEGACY_DEFAULT_MIGRATIONS).forEach(([key, [legacyValue, nextValue]]) => {
    if (migrated[key] === legacyValue) {
      migrated[key] = nextValue;
    }
  });
  return migrated;
}

function saveCurrentSettings() {
  const concurrentFiles = Number.parseInt(elements.concurrentFiles.value, 10);
  if (!Number.isInteger(concurrentFiles) || concurrentFiles < 1 || concurrentFiles > 3) return;

  const settings = {
    recordingSpeed: state.recordingSpeed,
    audioQuality: elements.audioQuality.value,
    transcriptionEngine: elements.transcriptionEngine.value,
    transcriptionQuality: elements.transcriptionQuality.value,
    whisperModel: elements.whisperModel.value,
    slideSensitivity: elements.slideSensitivity.value,
    enhanceWithGemini: state.folderMode === "processed" ? state.enhanceWithGeminiPreference : elements.enhanceWithGemini.checked,
    aiModel: elements.aiModel.value,
    concurrentFiles,
    saveNormalized: elements.saveNormalized.checked,
  };

  try {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Settings persistence is helpful, not required for processing.
  }
}

async function saveGeminiKey(options = {}) {
  const { quiet = false } = options;
  const key = elements.geminiApiKey.value.trim();
  if (!key) {
    if (!quiet) {
      elements.geminiKeyStatus.textContent = "Paste a key first";
    }
    return false;
  }

  elements.saveGeminiKeyButton.disabled = true;
  try {
    await invoke("save_api_key", { provider: "gemini", apiKey: key });
    elements.geminiApiKey.value = "";
    state.geminiKeySaved = true;
    elements.geminiKeyStatus.textContent = "Key saved";
    return true;
  } catch (error) {
    state.geminiKeySaved = false;
    elements.geminiKeyStatus.textContent = `Could not save key: ${error}`;
    return false;
  } finally {
    elements.saveGeminiKeyButton.disabled = false;
    renderAiControls();
  }
}

async function refreshGeminiKeyStatus() {
  try {
    const result = await invoke("has_api_key", { provider: "gemini" });
    state.geminiKeySaved = Boolean(result?.saved);
    elements.geminiKeyStatus.textContent = state.geminiKeySaved ? "Key saved" : "No key saved";
  } catch {
    state.geminiKeySaved = false;
    elements.geminiKeyStatus.textContent = "Key status unavailable";
  }
  renderAiControls();
}

function renderAiControls() {
  const geminiSelected = needsGemini();
  elements.aiModel.disabled = !geminiSelected;
  elements.geminiApiKey.disabled = !geminiSelected;
  elements.saveGeminiKeyButton.disabled = !geminiSelected;
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
  return Number.isInteger(parsed) && parsed >= 1 && parsed <= 3 ? parsed : DEFAULT_SETTINGS.concurrentFiles;
}

function renderNormalizationWarning() {
  elements.normalizationWarning.classList.toggle("hidden", !requiresNormalizationConfirmation());
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
  const ready = processableFileCount();
  const skipped = state.skippedFiles.size;
  const skippedText = skipped ? ` · ${skipped} skipped` : "";
  const noun = itemNoun();
  if (state.folderMode === "processed") {
    return `${ready} ${noun}${ready === 1 ? "" : "s"} · ${concurrent} at a time · Gemini${skippedText}`;
  }
  const aiText = needsGemini() ? " · Gemini" : "";
  return `${ready} ${noun}${ready === 1 ? "" : "s"} · ${concurrent} at a time · ${speed}${aiText}${skippedText}`;
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
  if (state.running) {
    renderVideoDashboard();
  }
}

function startSystemMetrics() {
  refreshSystemMetrics();
  state.systemMetricsTimer = window.setInterval(refreshSystemMetrics, 2500);
}

async function refreshSystemMetrics() {
  try {
    const metrics = await invoke("system_metrics");
    renderSystemMetrics(metrics);
  } catch {
    renderSystemMetrics(null);
  }
}

function renderSystemMetrics(metrics) {
  const cpuPercent = metrics?.cpuPercent;
  const gpuPercent = metrics?.gpuPercent;
  const memoryUsedGb = metrics?.memoryUsedGb;
  renderMetric(
    elements.cpuMetric,
    elements.cpuMetricStatus,
    cpuPercent == null ? "--" : `${Math.round(cpuPercent)}%`,
    cpuPercent == null ? metrics?.cpuStatus || "Unavailable" : "total capacity used",
    cpuPercent == null,
  );
  renderMetric(
    elements.gpuMetric,
    elements.gpuMetricStatus,
    gpuPercent == null ? "--" : `${Math.round(gpuPercent)}%`,
    gpuPercent == null ? metrics?.gpuStatus || "Unavailable" : "total capacity used",
    gpuPercent == null,
  );
  renderMetric(
    elements.memoryMetric,
    elements.memoryMetricStatus,
    memoryUsedGb == null ? "--" : `${Number(memoryUsedGb).toFixed(1)} GB`,
    memoryUsedGb == null ? metrics?.memoryStatus || "Unavailable" : "used memory in GB",
    memoryUsedGb == null,
  );
}

function renderMetric(valueElement, statusElement, value, status, unavailable) {
  valueElement.textContent = value;
  statusElement.textContent = status;
  valueElement.closest(".system-card")?.classList.toggle("unavailable", unavailable);
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
