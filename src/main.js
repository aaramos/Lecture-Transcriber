const invoke = window.__TAURI__?.core?.invoke;

if (!invoke) {
  throw new Error("Tauri API is not available. Start the app with `cargo tauri dev`.");
}

const STEP_LABELS = {
  Import: "Reading transcript",
  Probe: "Checking file",
  Normalize: "Normalizing",
  Audio: "Preparing audio",
  EnhanceAudio: "Enhancing audio",
  Transcribe: "Transcribing",
  Slides: "Extracting slides",
  Enrich: "Creating study notes",
  AIOverview: "AI overview",
  AITranscript: "AI transcript cleanup",
  AISlides: "AI slide analysis",
  AIResources: "AI resources",
  Render: "Building HTML",
};

const STEP_SHORT_LABELS = {
  Import: "Import",
  Probe: "Check",
  Normalize: "Normalize",
  Audio: "Audio",
  EnhanceAudio: "Clean",
  Transcribe: "Transcript",
  Slides: "Slides",
  Enrich: "AI",
  AIOverview: "Overview",
  AITranscript: "Text AI",
  AISlides: "Slide AI",
  AIResources: "Resources",
  Render: "HTML",
};

const AI_STEP_ORDER = ["AIOverview", "AITranscript", "AISlides", "AIResources"];
const STEP_ORDER = [
  "Import",
  "Probe",
  "Normalize",
  "Audio",
  "EnhanceAudio",
  "Transcribe",
  "Slides",
  "Enrich",
  ...AI_STEP_ORDER,
  "Render",
];

const STEP_START_PROGRESS = {
  Import: 18,
  Probe: 8,
  Normalize: 22,
  Audio: 42,
  EnhanceAudio: 50,
  Transcribe: 58,
  Slides: 82,
  Enrich: 88,
  AIOverview: 86,
  AITranscript: 89,
  AISlides: 92,
  AIResources: 94,
  Render: 96,
};

const STEP_DONE_PROGRESS = {
  Import: 72,
  Probe: 18,
  Normalize: 40,
  Audio: 48,
  EnhanceAudio: 56,
  Transcribe: 78,
  Slides: 96,
  Enrich: 94,
  AIOverview: 89,
  AITranscript: 92,
  AISlides: 94,
  AIResources: 96,
  Render: 100,
};

const SETTINGS_STORAGE_KEY = "lectureProcessor.settings.v2";
const LAST_OUTPUT_STORAGE_KEY = "lectureProcessor.lastOutputDir.v1";
const LOCAL_MODELS_STORAGE_KEY = "lectureProcessor.lmStudioModels.v2";
const MODEL_SELECTION_VERSION = 1;
const UI_SETTINGS_VERSION = 1;
const LM_STUDIO_OPENAI_URL = "http://192.168.86.101:1234/v1";
const GEMINI_RESOURCES_MODEL_VALUE = "__gemini_resources__";
const GEMINI_RESOURCE_PREFIX = "gemini:";
const GEMINI_MODELS = Object.freeze([
  { value: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
  { value: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash-Lite" },
]);
const LEGACY_LOCAL_MODEL_IDS = new Set(["gemma4:26b", "llama3", "qwen3", "gemma3"]);
const LEGACY_LOCAL_OPENAI_URLS = new Set([
  "http://localhost:11434/v1",
  "http://localhost:1234/v1",
  "http://localhost:8000/v1",
  "http://localhost:8001/v1",
]);
const DEFAULT_SETTINGS = Object.freeze({
  recordingSpeed: "1x",
  audioQuality: "high",
  audioEnhancement: "none",
  transcriptionProfile: "quality",
  slideSensitivity: "medium",
  aiModel: "gemini-2.5-flash",
  aiOverviewProvider: "mlx-text",
  aiOverviewModel: "",
  aiTranscriptProvider: "mlx-text",
  aiTranscriptModel: "",
  aiSlidesProvider: "mlx-vision",
  aiSlidesModel: "",
  aiResourcesProvider: "mlx-text",
  aiResourcesModel: "",
  mlxTextUrl: LM_STUDIO_OPENAI_URL,
  mlxVisionUrl: LM_STUDIO_OPENAI_URL,
  mlxTimeout: 120,
  enhanceWithGemini: false,
  concurrentFiles: 2,
  geminiMaxConcurrency: 3,
  saveNormalized: true,
});
const PROFILE_ORDER = ["quality", "fast", "turbo"];
const PROFILE_DESCRIPTIONS = Object.freeze({
  quality: "Highest accuracy. Best for difficult audio. ~1.5-2x real-time.",
  fast: "Recommended for most lectures. Minimal accuracy loss vs Quality, ~2-3x faster.",
  turbo: "Maximum speed using Apple Silicon acceleration. Excellent accuracy. Requires M-series Mac.",
});
const PROFILE_LABELS = Object.freeze({
  quality: "Quality",
  fast: "Fast",
  turbo: "Turbo",
});
const LEGACY_DEFAULT_MIGRATIONS = Object.freeze({
  audioQuality: ["fast", DEFAULT_SETTINGS.audioQuality],
  aiModel: ["gemini-2.5-flash-lite", DEFAULT_SETTINGS.aiModel],
});

const state = {
  folderMode: "source",
  inputDir: "",
  outputDir: "",
  fileCount: 0,
  fileNames: [],
  fileKinds: new Map(),
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
  tokensSent: 0,
  tokensReceived: 0,
  tokenUsageBySource: new Map(),
  geminiKeySaved: false,
  lmStudioTokenSaved: false,
  enhanceWithGeminiPreference: DEFAULT_SETTINGS.enhanceWithGemini,
  dependencyReady: false,
  dependencySetupRunning: false,
  profileStatusById: new Map(),
  files: new Map(),
  fileOrder: [],
  skippedFiles: new Set(),
  autoSkipReasons: new Map(),
  localModels: [],
  localModelsByBaseUrl: {},
  localModelErrorsByBaseUrl: {},
  localModelsLoading: false,
  localModelsLoadedFromCache: false,
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
  tokensSentMetric: document.querySelector("#tokensSentMetric"),
  tokensSentMetricStatus: document.querySelector("#tokensSentMetricStatus"),
  tokensReceivedMetric: document.querySelector("#tokensReceivedMetric"),
  tokensReceivedMetricStatus: document.querySelector("#tokensReceivedMetricStatus"),
  activeVideos: document.querySelector("#activeVideos"),
  activeSummary: document.querySelector("#activeSummary"),
  completedCount: document.querySelector("#completedCount"),
  processingCount: document.querySelector("#processingCount"),
  waitingCount: document.querySelector("#waitingCount"),
  failedCount: document.querySelector("#failedCount"),
  skippedCount: document.querySelector("#skippedCount"),
  canceledCount: document.querySelector("#canceledCount"),
  audioQuality: document.querySelector("#audioQuality"),
  audioEnhancement: document.querySelector("#audioEnhancement"),
  transcriptionProfile: document.querySelector("#transcriptionProfile"),
  transcriptionProfileHelp: document.querySelector("#transcriptionProfileHelp"),
  slideSensitivity: document.querySelector("#slideSensitivity"),
  enhanceWithGemini: document.querySelector("#enhanceWithGemini"),
  aiModel: document.querySelector("#aiModel"),
  aiOverviewProvider: document.querySelector("#aiOverviewProvider"),
  aiOverviewModel: document.querySelector("#aiOverviewModel"),
  aiTranscriptProvider: document.querySelector("#aiTranscriptProvider"),
  aiTranscriptModel: document.querySelector("#aiTranscriptModel"),
  aiSlidesProvider: document.querySelector("#aiSlidesProvider"),
  aiSlidesModel: document.querySelector("#aiSlidesModel"),
  aiResourcesProvider: document.querySelector("#aiResourcesProvider"),
  aiResourcesModel: document.querySelector("#aiResourcesModel"),
  mlxTextUrl: document.querySelector("#mlxTextUrl"),
  mlxVisionUrl: document.querySelector("#mlxVisionUrl"),
  mlxTimeout: document.querySelector("#mlxTimeout"),
  geminiApiKey: document.querySelector("#geminiApiKey"),
  saveGeminiKeyButton: document.querySelector("#saveGeminiKeyButton"),
  geminiKeyStatus: document.querySelector("#geminiKeyStatus"),
  lmStudioToken: document.querySelector("#lmStudioToken"),
  saveLmStudioTokenButton: document.querySelector("#saveLmStudioTokenButton"),
  lmStudioTokenStatus: document.querySelector("#lmStudioTokenStatus"),
  concurrentFiles: document.querySelector("#concurrentFiles"),
  geminiMaxConcurrency: document.querySelector("#geminiMaxConcurrency"),
  refreshLocalModelsButton: document.querySelector("#refreshLocalModelsButton"),
  localModelStatus: document.querySelector("#localModelStatus"),
  restoreDefaultsButton: document.querySelector("#restoreDefaultsButton"),
  speedSegments: [...document.querySelectorAll(".segment")],
};

window.__TAURI__?.event?.listen?.("processor-event", (event) => handleProcessorEvent(event.payload));
window.__TAURI__?.event?.listen?.("dependency-event", (event) => handleDependencyEvent(event.payload));
loadPersistedSettings();
loadLocalModelChoicesAtLaunch();
refreshDependencyStatus();
refreshTranscriptionProfileStatus();
refreshGeminiKeyStatus();
refreshLmStudioTokenStatus();
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
elements.refreshLocalModelsButton.addEventListener("click", () => refreshLocalModels({ force: true }));
elements.restoreDefaultsButton.addEventListener("click", restoreDefaultSettings);
elements.saveGeminiKeyButton.addEventListener("click", () => saveGeminiKey());
elements.saveLmStudioTokenButton.addEventListener("click", () => saveLmStudioToken());
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
elements.geminiMaxConcurrency.addEventListener("input", () => {
  validateSettings({ showDialogOnError: false });
  saveCurrentSettings();
});
elements.mlxTimeout.addEventListener("input", () => {
  validateSettings({ showDialogOnError: false });
  saveCurrentSettings();
});

[
  elements.audioQuality,
  elements.audioEnhancement,
  elements.transcriptionProfile,
  elements.slideSensitivity,
  elements.enhanceWithGemini,
  elements.aiOverviewModel,
  elements.aiSlidesModel,
  elements.aiResourcesModel,
].forEach((element) => {
  element.addEventListener("change", () => {
    syncAiRouteProvidersFromModelChoices();
    if (element === elements.enhanceWithGemini && state.folderMode !== "processed") {
      state.enhanceWithGeminiPreference = elements.enhanceWithGemini.checked;
    }
    if (element === elements.transcriptionProfile) {
      renderTranscriptionProfileHelp();
    }
    saveCurrentSettings();
    renderAiControls();
    render();
  });
});

[
  elements.mlxTextUrl,
  elements.mlxVisionUrl,
].forEach((element) => {
  element.addEventListener("input", () => {
    saveCurrentSettings();
    if (element === elements.mlxTextUrl || element === elements.mlxVisionUrl) {
      markLocalModelsStale();
    }
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
    const sourceFiles = Array.isArray(scan.sourceFiles) ? scan.sourceFiles : [];
    state.fileKinds = new Map(sourceFiles.map((file) => [file.name, file.kind || "video"]));
    state.fileNames = state.folderMode === "processed"
      ? scan.lectureFiles || []
      : sourceFiles.length
        ? sourceFiles.map((file) => file.name)
        : scan.movFiles || [];
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
        : "No supported lecture files found. Try a different folder or file.";
    setFolderError(state.fileCount === 0 ? emptyMessage : "");
  } catch (error) {
    state.folderMode = "source";
    state.fileCount = 0;
    state.fileNames = [];
    state.fileKinds.clear();
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
  state.fileKinds.clear();
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

  if (!ensureRequiredModelSelections()) return;

  if (selectedLocalRoutes().length && elements.lmStudioToken.value.trim()) {
    const saved = await saveLmStudioToken({ quiet: true });
    if (!saved) return;
  }

  const skipAiReason = await aiSkipReasonForUnavailableModels();
  const willRunAiEnhancement = needsAiEnhancement() && !skipAiReason;

  if (willRunAiEnhancement && needsGemini() && elements.geminiApiKey.value.trim()) {
    const saved = await saveGeminiKey({ quiet: true });
    if (!saved) return;
  }
  await refreshGeminiKeyStatus();
  await refreshLmStudioTokenStatus();
  const finalConcurrentFiles = validateSettings({ skipAiReason });
  if (!finalConcurrentFiles) return;
  const finalGeminiConcurrency = validGeminiConcurrency(elements.geminiMaxConcurrency.value);

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
    saveNormalizedVideo: true,
    audioQuality: elements.audioQuality.value,
    audioEnhancement: elements.audioEnhancement.value,
    transcriptionProfile: elements.transcriptionProfile.value,
    slideSensitivity: elements.slideSensitivity.value,
    aiProvider: needsAiEnhancement() ? "gemini" : "none",
    aiModel: selectedGeminiModel(),
    aiOverviewProvider: elements.aiOverviewProvider.value,
    aiOverviewModel: routeModelValue("overview"),
    aiTranscriptProvider: elements.aiTranscriptProvider.value,
    aiTranscriptModel: routeModelValue("transcript"),
    aiSlidesProvider: elements.aiSlidesProvider.value,
    aiSlidesModel: routeModelValue("slides"),
    aiResourcesProvider: elements.aiResourcesProvider.value,
    aiResourcesModel: routeModelValue("resources"),
    mlxTextUrl: elements.mlxTextUrl.value.trim(),
    mlxVisionUrl: elements.mlxVisionUrl.value.trim(),
    mlxTimeout: validMlxTimeout(elements.mlxTimeout.value),
    skipAiReason,
    geminiMaxConcurrency: finalGeminiConcurrency,
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
    markUnfinishedVideosFailed("Processor stopped before reporting this file complete.");
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
    await refreshTranscriptionProfileStatus();
  } catch (error) {
    showDependencySetup({
      message: "Dependency check failed",
      detail: String(error),
      progress: 0,
      failed: true,
    });
  }
}

async function refreshTranscriptionProfileStatus() {
  try {
    const status = await invoke("transcription_profile_status");
    state.profileStatusById = new Map((status?.profiles || []).map((profile) => [profile.id, profile]));
  } catch {
    state.profileStatusById = new Map();
  }
  renderTranscriptionProfileOptions();
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
    await refreshTranscriptionProfileStatus();
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
    elements.runMeta.textContent = "Stopping active file processing.";
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
    unfinished = markUnfinishedVideosFailed("Processor ended before reporting this file complete.");
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
    elements.runMeta.textContent = "All files finished. Open the output folder for transcripts, media, slides, and study pages.";
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
  elements.folderTitle.textContent = hasFolder ? basename(state.inputDir) : "Choose lecture file or folder";
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
      : "Select a file or folder to begin.";
  }
  renderVideoDashboard();
  renderAiControls();
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
  renderAiControls();
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
  state.tokensSent = 0;
  state.tokensReceived = 0;
  state.tokenUsageBySource.clear();
  state.files.clear();
  state.fileOrder = [];
  setProgress(0, 0);
  renderTokenMetrics();
  renderVideoDashboard();
}

function plannedStepNames(fileKind = "video") {
  if (state.folderMode === "processed") {
    return [...plannedAiStepNames(fileKind), "Render"];
  }

  if (fileKind === "transcript") {
    const steps = ["Import"];
    if (needsAiEnhancement()) {
      steps.push(...plannedAiStepNames(fileKind));
    }
    steps.push("Render");
    return steps;
  }

  const steps = ["Probe"];
  if (fileKind === "video" && state.recordingSpeed === "2x") {
    steps.push("Normalize");
  }
  steps.push("Audio", "Transcribe");
  if (fileKind === "video") {
    steps.push("Slides");
  }
  if (needsAiEnhancement()) {
    steps.push(...plannedAiStepNames(fileKind));
  }
  steps.push("Render");
  return steps;
}

function plannedAiStepNames(fileKind = "video") {
  const steps = ["AIOverview", "AITranscript"];
  if (fileKind === "video" || fileKind === "processed") {
    steps.push("AISlides");
  }
  steps.push("AIResources");
  return steps;
}

function buildInitialSteps(fileStatus = "queued", fileKind = "video") {
  const stepStatus = fileStatus === "skipped" ? "skipped" : "waiting";
  return plannedStepNames(fileKind).map((key) => ({
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

function aiStepKeyForProgress(step) {
  const normalized = String(step || "").toLowerCase();
  if (normalized.includes("overview")) return "AIOverview";
  if (normalized.includes("transcript")) return "AITranscript";
  if (normalized.includes("slide")) return "AISlides";
  if (normalized.includes("resource")) return "AIResources";
  return null;
}

function aiProgressDetail(event, aiStep, countText) {
  const label = aiStep ? STEP_LABELS[aiStep] : "AI notes";
  const model = String(event.model || "").trim();
  return `${label}${model ? ` · ${model}` : ""}${countText}`;
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
        if (item.status === "active") {
          return item;
        }
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
  const updatedSteps = steps.map((item) =>
    item.key === step
      ? {
          ...item,
          status: "done",
          elapsedSeconds,
          startedAt: item.startedAt || null,
        }
      : item,
  );
  const nextActiveStep = updatedSteps.find((item) => item.status === "active");
  const nextCurrentStep = file?.currentStep === step ? nextActiveStep?.key || null : file?.currentStep || nextActiveStep?.key || null;
  const nextCurrentStepState = updatedSteps.find((item) => item.key === nextCurrentStep);
  return {
    steps: updatedSteps,
    currentStep: nextCurrentStep,
    currentStepStartedAt: nextCurrentStepState?.startedAt || null,
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

function markAiStepStarted(file, step) {
  const now = Date.now();
  const steps = ensureStepStates(file, step);
  const activeIndex = steps.findIndex((item) => item.key === step);
  return {
    steps: steps.map((item, index) => {
      if (item.key === step) {
        return {
          ...item,
          status: "active",
          startedAt: item.startedAt || now,
          elapsedSeconds: null,
        };
      }
      if (index < activeIndex && isAiDetailStep(item.key) && ["waiting", "active"].includes(item.status)) {
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

function markAiStepsFinished(file, elapsedSeconds = null) {
  const steps = ensureStepStates(file);
  const now = Date.now();
  return {
    steps: steps.map((item) =>
      isAiDetailStep(item.key) && item.status !== "skipped" && item.status !== "stopped"
        ? {
            ...item,
            status: "done",
            elapsedSeconds: item.elapsedSeconds,
            startedAt: item.startedAt || null,
          }
        : item,
    ),
    currentStep: null,
    currentStepStartedAt: null,
    lastEventAt: now,
  };
}

function markAiStepsSkipped(file) {
  const steps = ensureStepStates(file);
  const now = Date.now();
  return {
    steps: steps.map((item) =>
      isAiDetailStep(item.key) && item.status !== "done"
        ? {
            ...item,
            status: "skipped",
            startedAt: item.startedAt || null,
          }
        : item,
    ),
    currentStep: null,
    currentStepStartedAt: null,
    lastEventAt: now,
  };
}

function progressFromSteps(steps, fallback = 0) {
  const visibleSteps = Array.isArray(steps) ? steps : [];
  if (!visibleSteps.length) return clampPercent(fallback);
  const units = visibleSteps.reduce((sum, step) => {
    if (step.status === "done" || step.status === "skipped") return sum + 1;
    if (step.status === "active") return sum + 0.5;
    return sum;
  }, 0);
  return clampPercent(Math.round((units / visibleSteps.length) * 100));
}

function ensureStepStates(file, step = null) {
  const existing = Array.isArray(file?.steps) && file.steps.length > 0
    ? file.steps
    : buildInitialSteps(file?.status, file?.sourceKind || "video");
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

function hasDetailedAiSteps(file) {
  return ensureStepStates(file).some((step) => isAiDetailStep(step.key));
}

function isAiDetailStep(step) {
  return AI_STEP_ORDER.includes(step);
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

function renderFileStats(file, source) {
  const length = formatFileLength(file?.durationSeconds);
  const elapsed = fileElapsedSeconds(file);
  const speed = processingSpeed(file?.durationSeconds, elapsed);
  const elapsedLabel = file?.status === "completed" ? "Finished in" : "Elapsed";
  const elapsedText = elapsed ? formatDuration(elapsed) : "--";
  const tokenStats = renderFileTokenStats(file, source);
  return `<div class="video-stats">Length ${escapeHtml(length)} · ${escapeHtml(elapsedLabel)} ${escapeHtml(
    elapsedText,
  )} · Speed ${escapeHtml(speed)}${tokenStats}</div>`;
}

function renderFileTokenStats(file, source) {
  const usage = state.tokenUsageBySource.get(source) || {
    sent: Number(file?.tokensSent || 0),
    received: Number(file?.tokensReceived || 0),
  };
  const hasUsage = Number(usage.sent) > 0 || Number(usage.received) > 0;
  const shouldShow = hasUsage || needsGemini() || file?.currentStep === "Enrich" || isAiDetailStep(file?.currentStep);
  if (!shouldShow) return "";
  const sent = hasUsage ? formatTokenCount(usage.sent) : "--";
  const received = hasUsage ? formatTokenCount(usage.received) : "--";
  return ` · Tokens in ${escapeHtml(sent)} · out ${escapeHtml(received)}`;
}

function fileElapsedSeconds(file) {
  if (!file) return 0;
  if (Number(file.elapsedSeconds) > 0) return Number(file.elapsedSeconds);
  if (file.startedAt && isActiveStatus(file.status)) {
    return Math.max(0, (Date.now() - file.startedAt) / 1000);
  }
  return 0;
}

function processingSpeed(durationSeconds, elapsedSeconds) {
  const duration = Number(durationSeconds || 0);
  const elapsed = Number(elapsedSeconds || 0);
  if (!duration || !elapsed || elapsed < 0.5) return "--x";
  return `${(duration / elapsed).toFixed(1)}x`;
}

function formatFileLength(seconds) {
  const totalSeconds = Number(seconds || 0);
  if (!totalSeconds || totalSeconds < 0) return "--:--";
  const wholeSeconds = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(wholeSeconds / 60);
  const remainingSeconds = wholeSeconds % 60;
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
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
  const { showDialogOnError = true, skipAiReason = "" } = options;
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
  const geminiValue = Number.parseInt(elements.geminiMaxConcurrency.value, 10);
  const geminiValid = Number.isInteger(geminiValue) && geminiValue >= 1 && geminiValue <= 12;
  elements.geminiMaxConcurrency.classList.toggle("invalid", !geminiValid);
  if (!geminiValid) {
    if (showDialogOnError) {
      showDialog(elements.settingsDialog);
    }
    elements.geminiMaxConcurrency.focus();
    elements.runMeta.textContent = "AI concurrency must be between 1 and 12.";
    return null;
  }
  const mlxTimeout = Number.parseInt(elements.mlxTimeout.value, 10);
  const mlxTimeoutValid = Number.isInteger(mlxTimeout) && mlxTimeout >= 10 && mlxTimeout <= 600;
  elements.mlxTimeout.classList.toggle("invalid", !mlxTimeoutValid);
  if (!mlxTimeoutValid) {
    if (showDialogOnError) {
      showDialog(elements.settingsDialog);
    }
    elements.mlxTimeout.focus();
    elements.runMeta.textContent = "MLX timeout must be between 10 and 600 seconds.";
    return null;
  }
  const selectedProfile = currentProfileStatus();
  if (selectedProfile && selectedProfile.available === false) {
    if (showDialogOnError) {
      showDialog(elements.settingsDialog);
    }
    elements.runMeta.textContent = selectedProfile.unavailableReason || "Choose an available transcription profile.";
    elements.transcriptionProfile.focus();
    return null;
  }
  if (!skipAiReason && needsGemini() && !state.geminiKeySaved && !elements.geminiApiKey.value.trim()) {
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
    const previous = state.files.get(event.source);
    updateFile(event.source, {
      sourceKind: event.source_kind || event.sourceKind || fileKindFor(event.source),
      status: "running",
      stage: "Preparing",
      detail: "Preparing file",
      progress: 5,
      startedAt: previous?.startedAt || Date.now(),
    });
  } else if (event.kind === "file_media") {
    const sourceKind = event.source_kind || event.sourceKind || (event.has_video === false ? "audio" : fileKindFor(event.source));
    updateFile(event.source, {
      sourceKind,
      durationSeconds: Number(event.duration_seconds || 0),
      sourceDurationSeconds: Number(event.source_duration_seconds || event.duration_seconds || 0),
    });
  } else if (event.kind === "step_started") {
    if (["skipped", "stopping", "stopped"].includes(state.files.get(event.source)?.status)) {
      return;
    }
    if (event.step === "Enrich" && hasDetailedAiSteps(state.files.get(event.source))) {
      const current = state.files.get(event.source);
      updateFile(event.source, {
        status: "running",
        stage: "Preparing AI",
        detail: "Waiting for AI overview",
        progress: progressFromSteps(current?.steps || [], STEP_START_PROGRESS.AIOverview),
        startedAt: current?.startedAt || Date.now(),
      });
      return;
    }
    const stepPatch = markStepStarted(state.files.get(event.source), event.step);
    updateFile(event.source, {
      status: "running",
      stage: STEP_LABELS[event.step] || event.step,
      detail: stepRunningDetail(event.step),
      progress: progressFromSteps(stepPatch.steps, STEP_START_PROGRESS[event.step] || 12),
      startedAt: state.files.get(event.source)?.startedAt || Date.now(),
      ...stepPatch,
    });
  } else if (event.kind === "step_finished") {
    if (["skipped", "stopping", "stopped"].includes(state.files.get(event.source)?.status)) {
      return;
    }
    if (event.step === "Enrich" && hasDetailedAiSteps(state.files.get(event.source))) {
      const current = state.files.get(event.source);
      const hasAiProgress = ensureStepStates(current).some((step) => isAiDetailStep(step.key) && step.status === "done");
      const aiPatch = hasAiProgress
        ? markAiStepsFinished(current, Number(event.elapsed_seconds || 0))
        : markAiStepsSkipped(current);
      updateFile(event.source, {
        stage: hasAiProgress ? "AI complete" : "AI skipped",
        detail: hasAiProgress ? `${Number(event.elapsed_seconds || 0).toFixed(1)}s` : "AI enhancement skipped",
        progress: progressFromSteps(aiPatch.steps, hasAiProgress ? STEP_DONE_PROGRESS.AIResources : STEP_START_PROGRESS.AIOverview),
        ...aiPatch,
      });
      return;
    }
    const stepPatch = markStepFinished(state.files.get(event.source), event.step, Number(event.elapsed_seconds || 0));
    const activeStep = stepPatch.steps.find((item) => item.status === "active");
    updateFile(event.source, {
      stage: activeStep ? STEP_LABELS[activeStep.key] || activeStep.key : `${STEP_LABELS[event.step] || event.step} complete`,
      detail: activeStep ? stepRunningDetail(activeStep.key) : `${Number(event.elapsed_seconds || 0).toFixed(1)}s`,
      progress: progressFromSteps(stepPatch.steps, STEP_DONE_PROGRESS[event.step] || 20),
      ...stepPatch,
    });
  } else if (event.kind === "enrichment_started") {
    const current = state.files.get(event.source);
    if (hasDetailedAiSteps(current)) {
      updateFile(event.source, {
        status: "running",
        stage: "Preparing AI",
        detail: "Waiting for AI overview",
        progress: progressFromSteps(current?.steps || [], STEP_START_PROGRESS.AIOverview),
      });
    } else {
      const enrichPatch = markStepStarted(current, "Enrich");
      updateFile(event.source, {
        detail: "Waiting for AI notes",
        progress: progressFromSteps(enrichPatch.steps, STEP_START_PROGRESS.Enrich),
        ...enrichPatch,
      });
    }
  } else if (event.kind === "enrichment_progress") {
    applyTokenUsage(event);
    const completed = Number(event.completed || 0);
    const total = Number(event.total || 0);
    const countText = total > 0 ? ` (${completed}/${total})` : "";
    const current = state.files.get(event.source);
    const aiStep = aiStepKeyForProgress(event.step);
    const activePatch = aiStep
      ? markAiStepStarted(current, aiStep)
      : current?.currentStep === "Enrich"
        ? {}
        : markStepStarted(current, "Enrich");
    const progressSteps = activePatch.steps || current?.steps || [];
    updateFile(event.source, {
      status: "running",
      stage: aiStep ? STEP_LABELS[aiStep] : STEP_LABELS.Enrich,
      detail: aiProgressDetail(event, aiStep, countText),
      progress: aiStep
        ? progressFromSteps(progressSteps, STEP_START_PROGRESS[aiStep])
        : Math.max(STEP_START_PROGRESS.Enrich, Math.min(STEP_DONE_PROGRESS.Enrich, 88 + completed)),
      ...activePatch,
    });
  } else if (event.kind === "enrichment_finished") {
    applyTokenUsage(event);
    const current = state.files.get(event.source);
    const aiPatch = hasDetailedAiSteps(current)
      ? markAiStepsFinished(current, Number(event.elapsed_seconds || 0))
      : markStepFinished(current, "Enrich", Number(event.elapsed_seconds || 0));
    updateFile(event.source, {
      detail: event.title ? `AI notes finished: ${event.title}` : "AI notes finished",
      progress: progressFromSteps(aiPatch.steps, STEP_DONE_PROGRESS.Enrich),
      ...aiPatch,
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
      durationSeconds: Number(event.lecture_duration_seconds || event.duration_seconds || previous.durationSeconds || 0),
      sourceDurationSeconds: Number(event.source_duration_seconds || previous.sourceDurationSeconds || 0),
      elapsedSeconds: Number(event.elapsed_seconds || previous.elapsedSeconds || 0),
      finishedAt: Date.now(),
      ...markFileFinished(previous, event.status, event.failure_step),
    });
  } else if (event.kind === "batch_finished") {
    setProgress(Number(event.attempted || state.progressDone), Number(event.attempted || state.progressTotal));
    renderVideoDashboard();
  } else if (event.kind === "batch_failed") {
    state.lastOutputDir = event.output_dir || state.outputDir;
    markUnfinishedVideosFailed(event.message || "Processor stopped before reporting this file complete.");
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
  const kind = patch.sourceKind || state.files.get(source)?.sourceKind || fileKindFor(source);
  const previous = state.files.get(source) || {
    status: "queued",
    stage: "Queue",
    detail: "Waiting to start",
    progress: 0,
    sourceKind: kind,
    steps: buildInitialSteps("queued", kind),
  };
  state.files.set(source, { ...previous, sourceKind: kind, ...patch });
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
      sourceKind: fileKindFor(name),
      status: skipped ? "skipped" : preparing ? "preparing" : "queued",
      stage: skipped ? "Skipped" : preparing ? "Preparing" : "Queue",
      detail: skipped ? skipReason : preparing ? `Preparing ${itemNoun()}` : "Waiting to start",
      progress: skipped ? 100 : preparing ? 3 : 0,
      steps: buildInitialSteps(skipped ? "skipped" : "queued", fileKindFor(name)),
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
      sourceKind: fileKindFor(name),
      status: skipped ? "skipped" : "queued",
      stage: skipped ? "Skipped" : "Queue",
      detail: skipped ? skipReason : "Waiting to start",
      progress: skipped ? 100 : 0,
      steps: buildInitialSteps(skipped ? "skipped" : "queued", fileKindFor(name)),
    });
  });
}

function syncPendingFilePlans() {
  if (state.finishedFileReports > 0) return;
  for (const [name, file] of state.files.entries()) {
    if (!["queued", "skipped", "preparing"].includes(file.status)) continue;
    state.files.set(name, {
      ...file,
      steps: buildInitialSteps(file.status === "skipped" ? "skipped" : "queued", file.sourceKind || fileKindFor(name)),
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
        sourceKind: fileKindFor(name),
        steps: buildInitialSteps("queued", fileKindFor(name)),
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
        sourceKind: fileKindFor(name),
        steps: buildInitialSteps("queued", fileKindFor(name)),
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
    elements.activeVideos.innerHTML = `<div class="video-placeholder">Choose a file or folder to see files.</div>`;
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
            ${renderFileStats(file, name)}
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
  if (summary.total === 0) return "Choose a file or folder to see files";
  const parts = [];
  if (summary.completed) parts.push(`${summary.completed} complete`);
  if (summary.processing) parts.push(`${summary.processing} active`);
  if (summary.waiting) parts.push(`${summary.waiting} queued`);
  if (summary.failed) parts.push(`${summary.failed} failed`);
  if (summary.skipped) parts.push(`${summary.skipped} skipped`);
  if (summary.stopped) parts.push(`${summary.stopped} stopped`);
  return parts.length ? parts.join(" · ") : "No files ready";
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
        steps: buildInitialSteps("queued", fileKindFor(source)),
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

function needsAiEnhancement() {
  return state.folderMode === "processed" || elements.enhanceWithGemini.checked;
}

function needsGemini() {
  return needsAiEnhancement() && usesGeminiRoutes();
}

function usesGeminiRoutes() {
  syncAiRouteProvidersFromModelChoices();
  return aiRouteProviders().some((provider) => provider === "gemini");
}

function aiRouteProviders() {
  return [
    elements.aiOverviewProvider.value,
    elements.aiTranscriptProvider.value,
    elements.aiSlidesProvider.value,
    elements.aiResourcesProvider.value,
  ];
}

function aiRouteControls() {
  return [
    { step: "overview", label: "Text Model", providerInput: elements.aiOverviewProvider, modelInput: elements.aiOverviewModel },
    { step: "transcript", label: "Text Model", providerInput: elements.aiTranscriptProvider, modelInput: elements.aiTranscriptModel },
    { step: "slides", label: "Vision Model", providerInput: elements.aiSlidesProvider, modelInput: elements.aiSlidesModel },
    { step: "resources", label: "Resource Model", providerInput: elements.aiResourcesProvider, modelInput: elements.aiResourcesModel },
  ];
}

function selectedLocalRoutes() {
  syncAiRouteProvidersFromModelChoices();
  if (!needsAiEnhancement()) return [];
  return aiRouteControls().filter((route) => isLocalOpenAiProvider(route.providerInput.value));
}

function isLocalOpenAiProvider(provider) {
  return provider === "mlx-text" || provider === "mlx-vision";
}

function syncAiRouteProvidersFromModelChoices() {
  elements.aiOverviewProvider.value = "mlx-text";
  elements.aiTranscriptProvider.value = "mlx-text";
  elements.aiTranscriptModel.value = routeModelValue("overview");
  elements.aiSlidesProvider.value = "mlx-vision";
  elements.aiResourcesProvider.value = isGeminiResourceValue(elements.aiResourcesModel.value) ? "gemini" : "mlx-text";
  elements.aiModel.value = selectedGeminiModel();
}

function routeModelValue(step) {
  const route = aiRouteControls().find((candidate) => candidate.step === step);
  if (!route) return "";
  if (step === "resources" && isGeminiResourceValue(route.modelInput.value)) return "";
  return String(route.modelInput.value || "").trim();
}

function storedRouteModelValue(step) {
  const route = aiRouteControls().find((candidate) => candidate.step === step);
  return route ? String(route.modelInput.value || "").trim() : "";
}

function geminiResourceValue(model) {
  return `${GEMINI_RESOURCE_PREFIX}${model || DEFAULT_SETTINGS.aiModel}`;
}

function isGeminiResourceValue(value) {
  const raw = String(value || "").trim();
  return raw === GEMINI_RESOURCES_MODEL_VALUE || raw.startsWith(GEMINI_RESOURCE_PREFIX);
}

function geminiModelFromResourceValue(value) {
  const raw = String(value || "").trim();
  if (raw === GEMINI_RESOURCES_MODEL_VALUE) return DEFAULT_SETTINGS.aiModel;
  if (raw.startsWith(GEMINI_RESOURCE_PREFIX)) {
    return raw.slice(GEMINI_RESOURCE_PREFIX.length) || DEFAULT_SETTINGS.aiModel;
  }
  return "";
}

function selectedGeminiModel() {
  return geminiModelFromResourceValue(elements.aiResourcesModel.value) || elements.aiModel.value || DEFAULT_SETTINGS.aiModel;
}

function resourceModelChoiceFromSettings(settings) {
  const raw = String(settings.aiResourcesModel || "").trim();
  if (raw === GEMINI_RESOURCES_MODEL_VALUE) return geminiResourceValue(settings.aiModel || DEFAULT_SETTINGS.aiModel);
  if (raw.startsWith(GEMINI_RESOURCE_PREFIX)) return raw;
  if (settings.aiResourcesProvider === "gemini") return geminiResourceValue(settings.aiModel || DEFAULT_SETTINGS.aiModel);
  return raw;
}

function ensureRequiredModelSelections() {
  const missingSelections = missingRequiredModelSelections();
  if (!missingSelections.length) return true;

  showDialog(elements.settingsDialog);
  const firstMissingRoute = aiRouteControls().find((route) => missingSelections.includes(route.label));
  firstMissingRoute?.modelInput.focus();
  elements.runMeta.textContent = `Choose ${missingSelections.join(", ")} before starting AI enhancement.`;
  return false;
}

function missingRequiredModelSelections() {
  if (!needsAiEnhancement()) return [];
  syncAiRouteProvidersFromModelChoices();
  return uniqueStrings(aiRouteControls()
    .filter((route) => {
      const value = storedRouteModelValue(route.step);
      return route.step === "resources" ? !value : !routeModelValue(route.step);
    })
    .map((route) => route.label));
}

async function aiSkipReasonForUnavailableModels() {
  if (!needsAiEnhancement()) return "";
  syncAiRouteProvidersFromModelChoices();

  // The Python pipeline now probes LM Studio once before staged AI starts and
  // skips only the unavailable role. Keep this refresh for user feedback/cache,
  // but do not turn one missing model into a whole-run AI skip here.
  await refreshLocalModels({ quiet: true, force: true });
  return "";
}

async function refreshLocalModels(options = {}) {
  const { quiet = false, force = false } = options;
  if (state.localModelsLoading) return state.localModels;
  if (!force && state.localModelsLoadedFromCache) return state.localModels;
  if (elements.lmStudioToken.value.trim()) {
    const saved = await saveLmStudioToken({ quiet: true });
    if (!saved) return state.localModels;
  }
  await refreshLmStudioTokenStatus();

  const baseUrls = localModelBaseUrls({ includeAll: true });
  state.localModelsLoading = true;
  renderAiControls();
  setLocalModelStatus("Checking models...");

  try {
    const results = await Promise.all(baseUrls.map(async (baseUrl) => {
      try {
        const models = uniqueStrings(await fetchLocalModels([baseUrl])).sort((left, right) => left.localeCompare(right));
        return { baseUrl, models, error: "" };
      } catch (error) {
        return { baseUrl, models: [], error: String(error || "Unavailable") };
      }
    }));
    const modelsByBaseUrl = {};
    const errorsByBaseUrl = {};
    results.forEach((result) => {
      modelsByBaseUrl[result.baseUrl] = result.models;
      if (result.error) {
        errorsByBaseUrl[result.baseUrl] = result.error;
      }
    });
    state.localModelsByBaseUrl = modelsByBaseUrl;
    state.localModelErrorsByBaseUrl = errorsByBaseUrl;
    state.localModels = uniqueStrings(results.flatMap((result) => result.models))
      .sort((left, right) => left.localeCompare(right));
    state.localModelsLoadedFromCache = false;
    cacheLocalModels();
    renderLocalModelOptions();
    setLocalModelStatus(localModelStatusSummary());
    if (!state.localModels.length && !quiet) {
      const errors = Object.values(errorsByBaseUrl);
      elements.runMeta.textContent = errors.length
        ? errors.join("; ")
        : "LM Studio is reachable, but no models are loaded.";
    }
    return state.localModels;
  } catch (error) {
    state.localModels = [];
    state.localModelsByBaseUrl = {};
    state.localModelErrorsByBaseUrl = {};
    state.localModelsLoadedFromCache = false;
    renderLocalModelOptions();
    setLocalModelStatus("LM Studio unavailable");
    if (!quiet) {
      const message = String(error || "");
      elements.runMeta.textContent = message.includes("API token")
        ? message
        : `Could not reach LM Studio at ${baseUrls.join(" and ")}.`;
    }
    return [];
  } finally {
    state.localModelsLoading = false;
    renderAiControls();
  }
}

function localModelBaseUrls(options = {}) {
  const { includeAll = false } = options;
  const urls = [];
  const routes = includeAll ? [] : selectedLocalRoutes();
  if (includeAll || routes.length === 0 || routes.some((route) => route.providerInput.value === "mlx-text")) {
    urls.push(normalizeOpenAiBaseUrl(elements.mlxTextUrl.value));
  }
  if (includeAll || routes.some((route) => route.providerInput.value === "mlx-vision")) {
    urls.push(normalizeOpenAiBaseUrl(elements.mlxVisionUrl.value));
  }
  return uniqueStrings(urls);
}

async function fetchLocalModels(baseUrls) {
  const result = await invoke("lm_studio_models", { baseUrls });
  return uniqueStrings(result?.models || []);
}

function renderLocalModelOptions() {
  renderModelSelect(elements.aiOverviewModel, routeModelValue("overview"), {
    includeGemini: false,
    models: modelsForRoute("overview"),
  });
  elements.aiTranscriptModel.value = routeModelValue("overview");
  renderModelSelect(elements.aiSlidesModel, routeModelValue("slides"), {
    includeGemini: false,
    models: modelsForRoute("slides"),
  });
  renderModelSelect(elements.aiResourcesModel, storedRouteModelValue("resources"), {
    includeGemini: true,
    models: modelsForRoute("resources"),
  });
  syncAiRouteProvidersFromModelChoices();
}

function renderModelSelect(select, selectedValue, options = {}) {
  const { includeGemini = false, models = state.localModels } = options;
  const modelOptions = uniqueStrings(models).sort((left, right) => left.localeCompare(right));
  const selected = String(selectedValue || "").trim();
  select.innerHTML = "";

  const placeholder = includeGemini ? "Choose Gemini or LM Studio model" : "Choose LM Studio model";
  select.appendChild(new Option(placeholder, ""));
  if (includeGemini) {
    GEMINI_MODELS.forEach((model) => {
      select.appendChild(new Option(model.label, geminiResourceValue(model.value)));
    });
  }
  modelOptions.forEach((model) => {
    select.appendChild(new Option(model, model));
  });

  if (selected && !isGeminiResourceValue(selected) && !modelOptions.includes(selected)) {
    select.appendChild(new Option(`${selected} (unavailable)`, selected));
  }
  if (selected) {
    select.value = selected;
  } else {
    select.value = "";
  }
}

function markLocalModelsStale() {
  state.localModels = [];
  state.localModelsByBaseUrl = {};
  state.localModelErrorsByBaseUrl = {};
  state.localModelsLoadedFromCache = false;
  renderLocalModelOptions();
  setLocalModelStatus("Models not checked");
}

function loadLocalModelChoicesAtLaunch() {
  if (loadCachedLocalModels()) {
    renderLocalModelOptions();
    setLocalModelStatus(localModelStatusSummary({ cached: true }));
    saveCurrentSettings();
    return;
  }
  refreshLocalModels({ quiet: true, force: true }).then(() => saveCurrentSettings());
}

function loadCachedLocalModels() {
  try {
    const cached = JSON.parse(window.localStorage.getItem(LOCAL_MODELS_STORAGE_KEY) || "{}");
    const cachedUrls = Array.isArray(cached.baseUrls) ? cached.baseUrls : [];
    if (JSON.stringify(cachedUrls) !== JSON.stringify(localModelBaseUrls({ includeAll: true }))) return false;
    if (!cached.modelsByBaseUrl || typeof cached.modelsByBaseUrl !== "object") return false;
    state.localModelsByBaseUrl = Object.fromEntries(
      Object.entries(cached.modelsByBaseUrl).map(([baseUrl, models]) => [
        normalizeOpenAiBaseUrl(baseUrl),
        uniqueStrings(Array.isArray(models) ? models : []),
      ])
    );
    state.localModelErrorsByBaseUrl = {};
    state.localModels = uniqueStrings(Object.values(state.localModelsByBaseUrl).flat())
      .sort((left, right) => left.localeCompare(right));
    state.localModelsLoadedFromCache = true;
    return state.localModels.length > 0;
  } catch {
    return false;
  }
}

function cacheLocalModels() {
  try {
    window.localStorage.setItem(
      LOCAL_MODELS_STORAGE_KEY,
      JSON.stringify({
        baseUrls: localModelBaseUrls({ includeAll: true }),
        modelsByBaseUrl: state.localModelsByBaseUrl,
        updatedAt: new Date().toISOString(),
      })
    );
  } catch {
    // Model cache only improves startup; live refresh still works without it.
  }
}

function setLocalModelStatus(message) {
  elements.localModelStatus.textContent = message;
}

function modelsForRoute(step) {
  if (step === "slides") return modelsForBaseUrl(elements.mlxVisionUrl.value);
  return modelsForBaseUrl(elements.mlxTextUrl.value);
}

function modelsForBaseUrl(baseUrl) {
  return state.localModelsByBaseUrl[normalizeOpenAiBaseUrl(baseUrl)] || [];
}

function localModelStatusSummary(options = {}) {
  const { cached = false } = options;
  const textUrl = normalizeOpenAiBaseUrl(elements.mlxTextUrl.value);
  const visionUrl = normalizeOpenAiBaseUrl(elements.mlxVisionUrl.value);
  const prefix = cached ? "Cached: " : "";
  if (textUrl === visionUrl) {
    return `${prefix}Shared server ${serverModelStatus(textUrl)}`;
  }
  return `${prefix}Text/Resource ${serverModelStatus(textUrl)} · Vision ${serverModelStatus(visionUrl)}`;
}

function serverModelStatus(baseUrl) {
  const models = modelsForBaseUrl(baseUrl);
  if (state.localModelErrorsByBaseUrl[normalizeOpenAiBaseUrl(baseUrl)]) return "unavailable";
  return `${models.length} model${models.length === 1 ? "" : "s"}`;
}

function normalizeOpenAiBaseUrl(value) {
  const raw = String(value || LM_STUDIO_OPENAI_URL).trim() || LM_STUDIO_OPENAI_URL;
  try {
    const url = new URL(raw);
    const path = url.pathname.replace(/\/+$/, "");
    url.pathname = path && path !== "/" ? path : "/v1";
    url.search = "";
    url.hash = "";
    return url.toString().replace(/\/+$/, "");
  } catch {
    return raw.replace(/\/+$/, "");
  }
}

function uniqueStrings(values) {
  return [...new Set(values.map((value) => String(value || "").trim()).filter(Boolean))];
}

function requiresNormalizationConfirmation() {
  return state.folderMode !== "processed" && state.recordingSpeed === "2x";
}

function itemNoun() {
  return state.folderMode === "processed" ? "lecture" : "file";
}

function fileKindFor(source) {
  return state.folderMode === "processed" ? "processed" : state.fileKinds.get(source) || "video";
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
  setSelectValue(elements.audioEnhancement, settings.audioEnhancement, DEFAULT_SETTINGS.audioEnhancement);
  setSelectValue(elements.transcriptionProfile, settings.transcriptionProfile, DEFAULT_SETTINGS.transcriptionProfile);
  setSelectValue(elements.slideSensitivity, settings.slideSensitivity, DEFAULT_SETTINGS.slideSensitivity);
  setSelectValue(elements.aiModel, settings.aiModel, DEFAULT_SETTINGS.aiModel);
  elements.mlxTextUrl.value = settings.mlxTextUrl || DEFAULT_SETTINGS.mlxTextUrl;
  elements.mlxVisionUrl.value = settings.mlxVisionUrl || DEFAULT_SETTINGS.mlxVisionUrl;
  elements.mlxTimeout.value = String(validMlxTimeout(settings.mlxTimeout));
  const resourceChoice = resourceModelChoiceFromSettings(settings);
  elements.aiOverviewProvider.value = "mlx-text";
  elements.aiTranscriptProvider.value = "mlx-text";
  elements.aiSlidesProvider.value = "mlx-vision";
  elements.aiResourcesProvider.value = isGeminiResourceValue(resourceChoice) ? "gemini" : "mlx-text";
  renderLocalModelOptions();
  renderModelSelect(elements.aiOverviewModel, settings.aiOverviewModel || DEFAULT_SETTINGS.aiOverviewModel, {
    includeGemini: false,
    models: modelsForRoute("overview"),
  });
  elements.aiTranscriptModel.value = routeModelValue("overview");
  renderModelSelect(elements.aiSlidesModel, settings.aiSlidesModel || DEFAULT_SETTINGS.aiSlidesModel, {
    includeGemini: false,
    models: modelsForRoute("slides"),
  });
  renderModelSelect(elements.aiResourcesModel, resourceChoice || DEFAULT_SETTINGS.aiResourcesModel, {
    includeGemini: true,
    models: modelsForRoute("resources"),
  });
  syncAiRouteProvidersFromModelChoices();
  state.enhanceWithGeminiPreference = Boolean(settings.enhanceWithGemini);
  elements.enhanceWithGemini.checked = state.enhanceWithGeminiPreference;
  elements.concurrentFiles.value = String(validConcurrentFiles(settings.concurrentFiles));
  elements.geminiMaxConcurrency.value = String(validGeminiConcurrency(settings.geminiMaxConcurrency));
  syncSpeedSegments();
  renderTranscriptionProfileOptions();
  renderAiControls();
}

function migratePersistedSettings(settings) {
  const migrated = { ...settings };
  if (!migrated.transcriptionProfile) {
    migrated.transcriptionProfile = legacyProfileFromSettings(migrated);
  }
  if (LEGACY_LOCAL_OPENAI_URLS.has(normalizeOpenAiBaseUrl(migrated.mlxTextUrl))) {
    migrated.mlxTextUrl = DEFAULT_SETTINGS.mlxTextUrl;
  }
  if (LEGACY_LOCAL_OPENAI_URLS.has(normalizeOpenAiBaseUrl(migrated.mlxVisionUrl))) {
    migrated.mlxVisionUrl = DEFAULT_SETTINGS.mlxVisionUrl;
  }
  if (migrated.modelSelectionVersion !== MODEL_SELECTION_VERSION) {
    migrated.aiOverviewModel = "";
    migrated.aiTranscriptModel = "";
    migrated.aiSlidesModel = "";
    migrated.aiResourcesModel = "";
    migrated.aiResourcesProvider = DEFAULT_SETTINGS.aiResourcesProvider;
  }
  if (migrated.uiSettingsVersion !== UI_SETTINGS_VERSION) {
    migrated.geminiMaxConcurrency = DEFAULT_SETTINGS.geminiMaxConcurrency;
    migrated.saveNormalized = true;
    if (migrated.aiResourcesProvider === "gemini" || migrated.aiResourcesModel === GEMINI_RESOURCES_MODEL_VALUE) {
      migrated.aiResourcesModel = geminiResourceValue(migrated.aiModel || DEFAULT_SETTINGS.aiModel);
      migrated.aiResourcesProvider = "gemini";
    }
  }
  ["aiOverviewModel", "aiTranscriptModel", "aiSlidesModel", "aiResourcesModel"].forEach((key) => {
    if (LEGACY_LOCAL_MODEL_IDS.has(String(migrated[key] || "").trim())) {
      migrated[key] = "";
    }
  });
  Object.entries(LEGACY_DEFAULT_MIGRATIONS).forEach(([key, [legacyValue, nextValue]]) => {
    if (migrated[key] === legacyValue) {
      migrated[key] = nextValue;
    }
  });
  return migrated;
}

function legacyProfileFromSettings(settings) {
  const legacyQuality = String(settings.transcriptionQuality || "").toLowerCase();
  if (legacyQuality === "balanced" || legacyQuality === "fast") return "fast";
  return DEFAULT_SETTINGS.transcriptionProfile;
}

function saveCurrentSettings() {
  syncAiRouteProvidersFromModelChoices();
  const concurrentFiles = Number.parseInt(elements.concurrentFiles.value, 10);
  if (!Number.isInteger(concurrentFiles) || concurrentFiles < 1 || concurrentFiles > 3) return;
  const geminiMaxConcurrency = Number.parseInt(elements.geminiMaxConcurrency.value, 10);
  if (!Number.isInteger(geminiMaxConcurrency) || geminiMaxConcurrency < 1 || geminiMaxConcurrency > 12) return;
  const mlxTimeout = Number.parseInt(elements.mlxTimeout.value, 10);
  if (!Number.isInteger(mlxTimeout) || mlxTimeout < 10 || mlxTimeout > 600) return;

  const settings = {
    uiSettingsVersion: UI_SETTINGS_VERSION,
    modelSelectionVersion: MODEL_SELECTION_VERSION,
    recordingSpeed: state.recordingSpeed,
    audioQuality: elements.audioQuality.value,
    audioEnhancement: elements.audioEnhancement.value,
    transcriptionProfile: elements.transcriptionProfile.value,
    slideSensitivity: elements.slideSensitivity.value,
    enhanceWithGemini: state.folderMode === "processed" ? state.enhanceWithGeminiPreference : elements.enhanceWithGemini.checked,
    aiModel: selectedGeminiModel(),
    aiOverviewProvider: elements.aiOverviewProvider.value,
    aiOverviewModel: routeModelValue("overview"),
    aiTranscriptProvider: elements.aiTranscriptProvider.value,
    aiTranscriptModel: routeModelValue("transcript"),
    aiSlidesProvider: elements.aiSlidesProvider.value,
    aiSlidesModel: routeModelValue("slides"),
    aiResourcesProvider: elements.aiResourcesProvider.value,
    aiResourcesModel: storedRouteModelValue("resources"),
    mlxTextUrl: elements.mlxTextUrl.value.trim(),
    mlxVisionUrl: elements.mlxVisionUrl.value.trim(),
    mlxTimeout,
    concurrentFiles,
    geminiMaxConcurrency,
    saveNormalized: true,
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

async function saveLmStudioToken(options = {}) {
  const { quiet = false } = options;
  const token = elements.lmStudioToken.value.trim();
  if (!token) {
    if (!quiet) {
      elements.lmStudioTokenStatus.textContent = "Paste a token first";
    }
    return false;
  }

  elements.saveLmStudioTokenButton.disabled = true;
  try {
    await invoke("save_api_key", { provider: "lm-studio", apiKey: token });
    elements.lmStudioToken.value = "";
    state.lmStudioTokenSaved = true;
    elements.lmStudioTokenStatus.textContent = "Token saved";
    return true;
  } catch (error) {
    state.lmStudioTokenSaved = false;
    elements.lmStudioTokenStatus.textContent = `Could not save token: ${error}`;
    return false;
  } finally {
    elements.saveLmStudioTokenButton.disabled = false;
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

async function refreshLmStudioTokenStatus() {
  try {
    const result = await invoke("has_api_key", { provider: "lm-studio" });
    state.lmStudioTokenSaved = Boolean(result?.saved);
    elements.lmStudioTokenStatus.textContent = state.lmStudioTokenSaved ? "Token saved" : "No token saved";
  } catch {
    state.lmStudioTokenSaved = false;
    elements.lmStudioTokenStatus.textContent = "Token status unavailable";
  }
  renderAiControls();
}

function renderAiControls() {
  const geminiSelected = needsGemini();
  elements.geminiApiKey.disabled = !geminiSelected;
  elements.saveGeminiKeyButton.disabled = !geminiSelected;
  elements.lmStudioToken.disabled = state.running;
  elements.saveLmStudioTokenButton.disabled = state.running;
  [
    elements.aiOverviewProvider,
    elements.aiOverviewModel,
    elements.aiTranscriptProvider,
    elements.aiTranscriptModel,
    elements.aiSlidesProvider,
    elements.aiSlidesModel,
    elements.aiResourcesProvider,
    elements.aiResourcesModel,
    elements.mlxTextUrl,
    elements.mlxVisionUrl,
    elements.mlxTimeout,
    elements.refreshLocalModelsButton,
  ].forEach((element) => {
    const waitsForLocalModels = element === elements.refreshLocalModelsButton;
    element.disabled = state.running || (waitsForLocalModels && state.localModelsLoading);
  });
  elements.refreshLocalModelsButton.textContent = state.localModelsLoading ? "Checking..." : "Force Refresh Models";
}

function renderTranscriptionProfileOptions() {
  const currentValue = elements.transcriptionProfile.value || DEFAULT_SETTINGS.transcriptionProfile;
  const profiles = PROFILE_ORDER.map((id) => state.profileStatusById.get(id) || {
    id,
    displayName: PROFILE_LABELS[id],
    description: PROFILE_DESCRIPTIONS[id],
    available: true,
    unavailableReason: null,
  });
  elements.transcriptionProfile.innerHTML = "";
  profiles.forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.displayName || PROFILE_LABELS[profile.id] || profile.id;
    option.disabled = profile.available === false;
    if (profile.unavailableReason) {
      option.title = profile.unavailableReason;
    }
    elements.transcriptionProfile.appendChild(option);
  });
  const desiredStatus = profiles.find((profile) => profile.id === currentValue);
  elements.transcriptionProfile.value = desiredStatus && desiredStatus.available !== false
    ? currentValue
    : DEFAULT_SETTINGS.transcriptionProfile;
  renderTranscriptionProfileHelp();
}

function renderTranscriptionProfileHelp() {
  const profile = currentProfileStatus();
  if (!profile) {
    elements.transcriptionProfileHelp.textContent = PROFILE_DESCRIPTIONS[elements.transcriptionProfile.value] || "";
    return;
  }
  const reason = profile.available === false && profile.unavailableReason ? ` ${profile.unavailableReason}` : "";
  elements.transcriptionProfileHelp.textContent =
    `${profile.description || PROFILE_DESCRIPTIONS[profile.id] || ""}${reason}`;
}

function currentProfileStatus() {
  const id = elements.transcriptionProfile.value || DEFAULT_SETTINGS.transcriptionProfile;
  return state.profileStatusById.get(id) || {
    id,
    displayName: PROFILE_LABELS[id],
    description: PROFILE_DESCRIPTIONS[id],
    available: true,
    unavailableReason: null,
  };
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

function validGeminiConcurrency(value) {
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= 1 && parsed <= 12 ? parsed : DEFAULT_SETTINGS.geminiMaxConcurrency;
}

function validMlxTimeout(value) {
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= 10 && parsed <= 600 ? parsed : DEFAULT_SETTINGS.mlxTimeout;
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
    return `${ready} ${noun}${ready === 1 ? "" : "s"} · ${concurrent} at a time · ${selectedAiRouteLabel()}${skippedText}`;
  }
  const profileText = selectedProfileLabel();
  const audioText = selectedAudioEnhancementLabel();
  const aiText = needsAiEnhancement() ? ` · ${selectedAiRouteLabel()}` : "";
  return `${ready} ${noun}${ready === 1 ? "" : "s"} · ${concurrent} at a time · ${speed} · ${profileText}${audioText}${aiText}${skippedText}`;
}

function selectedProfileLabel() {
  const profile = currentProfileStatus();
  return profile?.displayName || PROFILE_LABELS[elements.transcriptionProfile.value] || "Quality";
}

function selectedAudioEnhancementLabel() {
  const value = elements.audioEnhancement.value || DEFAULT_SETTINGS.audioEnhancement;
  if (value === "hybrid") return " · Hybrid audio";
  if (value === "strong") return " · Strong audio";
  return "";
}

function selectedAiRouteLabel() {
  const providers = aiRouteProviders();
  if (providers.every((provider) => provider === "gemini")) return "Gemini";
  if (providers.every((provider) => provider === "off")) return "AI off";
  if (!providers.includes("gemini")) return "LM Studio";
  return "Gemini + LM Studio";
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
  renderTokenMetrics();
}

function renderMetric(valueElement, statusElement, value, status, unavailable) {
  valueElement.textContent = value;
  statusElement.textContent = status;
  valueElement.closest(".system-card")?.classList.toggle("unavailable", unavailable);
}

function applyTokenUsage(event) {
  const sent = Number(event.input_tokens ?? event.inputTokens ?? event.input_token_estimate ?? event.inputTokenEstimate);
  const received = Number(event.output_tokens ?? event.outputTokens ?? event.output_token_estimate ?? event.outputTokenEstimate);
  if (!Number.isFinite(sent) && !Number.isFinite(received)) return;

  if (event.source) {
    const previous = state.tokenUsageBySource.get(event.source) || { sent: 0, received: 0 };
    state.tokenUsageBySource.set(event.source, {
      sent: Number.isFinite(sent) ? Math.max(0, sent) : previous.sent,
      received: Number.isFinite(received) ? Math.max(0, received) : previous.received,
    });
    const totals = [...state.tokenUsageBySource.values()].reduce(
      (sum, usage) => ({
        sent: sum.sent + usage.sent,
        received: sum.received + usage.received,
      }),
      { sent: 0, received: 0 },
    );
    state.tokensSent = totals.sent;
    state.tokensReceived = totals.received;
  } else {
    if (Number.isFinite(sent)) state.tokensSent = Math.max(0, sent);
    if (Number.isFinite(received)) state.tokensReceived = Math.max(0, received);
  }

  renderTokenMetrics();
}

function renderTokenMetrics() {
  elements.tokensSentMetric.textContent = formatTokenCount(state.tokensSent);
  elements.tokensReceivedMetric.textContent = formatTokenCount(state.tokensReceived);
  elements.tokensSentMetricStatus.textContent = state.tokensSent ? "AI prompt" : "Waiting";
  elements.tokensReceivedMetricStatus.textContent = state.tokensReceived ? "AI output" : "Waiting";
}

function formatTokenCount(value) {
  const count = Math.max(0, Math.round(Number(value) || 0));
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 10_000) return `${Math.round(count / 1_000)}K`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K`;
  return String(count);
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
