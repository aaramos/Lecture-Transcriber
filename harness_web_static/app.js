const state = {
  session: null,
  tests: [],
  activeTestId: "",
  activeTest: null,
  profiles: [],
  prompts: [],
  models: [],
  selectedModels: new Set(),
  activeJobId: null,
  pollTimer: null,
  resultsRuns: [],
  resultsSort: {
    key: "run_id",
    direction: "asc",
  },
};

const JOB_STORAGE_KEY = "visionHarnessActiveJobId";

const els = {
  testSelect: document.querySelector("#testSelect"),
  newTest: document.querySelector("#newTest"),
  testStatus: document.querySelector("#testStatus"),
  newTestPanel: document.querySelector("#newTestPanel"),
  newTestName: document.querySelector("#newTestName"),
  newTestPrompt: document.querySelector("#newTestPrompt"),
  newTestDetail: document.querySelector("#newTestDetail"),
  cancelNewTest: document.querySelector("#cancelNewTest"),
  createNewTest: document.querySelector("#createNewTest"),
  baseUrl: document.querySelector("#baseUrl"),
  apiToken: document.querySelector("#apiToken"),
  saveToken: document.querySelector("#saveToken"),
  tokenStatus: document.querySelector("#tokenStatus"),
  folderInput: document.querySelector("#folderInput"),
  dropZone: document.querySelector("#dropZone"),
  imageCount: document.querySelector("#imageCount"),
  imageList: document.querySelector("#imageList"),
  refreshModels: document.querySelector("#refreshModels"),
  showAllModels: document.querySelector("#showAllModels"),
  modelList: document.querySelector("#modelList"),
  profileSelect: document.querySelector("#profileSelect"),
  profileDetail: document.querySelector("#profileDetail"),
  promptSelect: document.querySelector("#promptSelect"),
  promptName: document.querySelector("#promptName"),
  promptText: document.querySelector("#promptText"),
  promptHash: document.querySelector("#promptHash"),
  editPrompt: document.querySelector("#editPrompt"),
  savePrompt: document.querySelector("#savePrompt"),
  savePromptAs: document.querySelector("#savePromptAs"),
  deletePrompt: document.querySelector("#deletePrompt"),
  readyState: document.querySelector("#readyState"),
  readyDetail: document.querySelector("#readyDetail"),
  startRun: document.querySelector("#startRun"),
  jobState: document.querySelector("#jobState"),
  progressBar: document.querySelector("#progressBar"),
  currentModel: document.querySelector("#currentModel"),
  currentImage: document.querySelector("#currentImage"),
  logList: document.querySelector("#logList"),
  reloadResults: document.querySelector("#reloadResults"),
  clearResults: document.querySelector("#clearResults"),
  resultsSummary: document.querySelector("#resultsSummary"),
  resultsTable: document.querySelector("#resultsTable tbody"),
  reportPreview: document.querySelector("#reportPreview"),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const message = typeof payload === "object" && payload.error ? payload.error : response.statusText;
    throw new Error(message);
  }
  return payload;
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === name);
  });
}

function enableResultsTab() {
  document.querySelector('.tab[data-tab="results"]').disabled = false;
}

function setMessage(kind, detail) {
  els.readyState.textContent = kind;
  els.readyDetail.textContent = detail;
}

function updateReadyState() {
  const hasImages = Boolean(state.session);
  const hasModels = state.selectedModels.size > 0;
  const hasPrompt = Boolean(els.promptSelect.value);
  const hasProfile = Boolean(els.profileSelect.value);
  els.startRun.disabled = !(hasImages && hasModels && hasPrompt && hasProfile);
  if (!hasImages) {
    setMessage("Waiting for images.", "Choose or drag a folder first.");
  } else if (!hasModels) {
    setMessage("Waiting for model selection.", "Choose one or more detected vision models.");
  } else if (!hasPrompt) {
    setMessage("Waiting for prompt.", "Choose a prompt before running.");
  } else if (!hasProfile) {
    setMessage("Waiting for profile.", "Choose an inference profile before running.");
  } else {
    setMessage("Ready to run.", `${state.session.images.length} images across ${state.selectedModels.size} model(s).`);
  }
}

async function loadConfig() {
  const config = await api("/api/config");
  els.baseUrl.value = config.base_url;
  renderTokenStatus(config.token || {});
  applyTestState(config);
  state.profiles = config.profiles || [];
  renderProfiles();
  state.prompts = config.prompts;
  renderPrompts();
  if (config.session) {
    state.session = config.session;
    renderImages(config.session);
  } else {
    clearImages();
  }
  if (config.has_results) {
    enableResultsTab();
    await loadResults();
  }
  await recoverJobLog();
}

function applyTestState(payload) {
  state.tests = payload.tests || [];
  state.activeTestId = payload.active_test_id || "";
  state.activeTest = payload.active_test || state.tests.find((test) => test.id === state.activeTestId) || null;
  renderTests();
}

function renderTests() {
  els.testSelect.innerHTML = "";
  state.tests.forEach((test) => {
    const option = document.createElement("option");
    option.value = test.id;
    option.textContent = test.name || test.id;
    els.testSelect.append(option);
  });
  els.testSelect.value = state.activeTestId;
  const active = state.activeTest || {};
  const pieces = [];
  pieces.push(`${active.run_count || 0} run${active.run_count === 1 ? "" : "s"}`);
  pieces.push(`${active.baseline_count || 0} baseline verdict${active.baseline_count === 1 ? "" : "s"}`);
  if (active.image_count) pieces.push(`${active.image_count} images`);
  els.testStatus.textContent = pieces.join(" · ");
}

function renderProfiles() {
  const requested = state.activeTest && state.activeTest.default_profile
    ? state.activeTest.default_profile
    : els.profileSelect.value;
  els.profileSelect.innerHTML = "";
  state.profiles.forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.name;
    option.textContent = profile.name;
    els.profileSelect.append(option);
  });
  if (requested && state.profiles.some((profile) => profile.name === requested)) {
    els.profileSelect.value = requested;
  }
  renderSelectedProfile();
}

function renderSelectedProfile() {
  const profile = state.profiles.find((item) => item.name === els.profileSelect.value);
  if (!profile) {
    els.profileDetail.textContent = "No inference profile selected.";
    updateReadyState();
    return;
  }
  const inference = profile.inference || {};
  const load = profile.load || {};
  els.profileDetail.textContent = [
    `Load: context ${load.context_length || "default"}, eval batch ${load.eval_batch_size || "default"}`,
    `Inference: max ${inference.max_tokens || "default"} tokens, temperature ${inference.temperature ?? "default"}, top_k ${inference.top_k ?? "default"}`,
  ].join("\n");
  updateReadyState();
}

async function persistProfile() {
  const profile = els.profileSelect.value;
  if (!profile || !state.activeTestId) return;
  const payload = await api("/api/tests/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ test_id: state.activeTestId, profile }),
  });
  applyTestState(payload);
  renderProfiles();
}

function renderTokenStatus(token) {
  if (token.saved) {
    els.tokenStatus.textContent = token.source === "environment"
      ? "Token available from environment"
      : `Token saved in ${token.keychain_service || "Keychain"}`;
  } else {
    els.tokenStatus.textContent = "No token saved";
  }
}

async function saveToken() {
  const token = els.apiToken.value.trim();
  if (!token) {
    els.tokenStatus.textContent = "Paste a token first";
    return;
  }
  els.saveToken.disabled = true;
  els.tokenStatus.textContent = "Saving token...";
  try {
    const payload = await api("/api/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    els.apiToken.value = "";
    renderTokenStatus(payload.token || {});
  } finally {
    els.saveToken.disabled = false;
  }
}

function renderPrompts() {
  const requested = state.activeTest && state.activeTest.default_prompt
    ? state.activeTest.default_prompt
    : els.promptSelect.value;
  els.promptSelect.innerHTML = "";
  state.prompts.forEach((prompt) => {
    const option = document.createElement("option");
    option.value = prompt.name;
    option.textContent = prompt.name;
    els.promptSelect.append(option);
  });
  if (requested && state.prompts.some((prompt) => prompt.name === requested)) {
    els.promptSelect.value = requested;
  }
  renderSelectedPrompt();
}

function renderSelectedPrompt() {
  const prompt = state.prompts.find((item) => item.name === els.promptSelect.value);
  els.promptText.value = prompt ? prompt.text : "";
  els.promptName.value = prompt ? prompt.name : "";
  els.promptText.readOnly = true;
  els.editPrompt.hidden = false;
  els.savePrompt.hidden = true;
  els.savePromptAs.hidden = true;
  els.promptHash.textContent = prompt ? `Prompt hash: ${prompt.hash}` : "";
  updateReadyState();
}

function enablePromptEditing() {
  els.promptText.readOnly = false;
  els.editPrompt.hidden = true;
  els.savePrompt.hidden = false;
  els.savePromptAs.hidden = false;
  els.promptText.focus();
}

async function savePrompt() {
  const name = els.promptSelect.value;
  const payload = await api(`/api/prompts/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: els.promptText.value }),
  });
  const index = state.prompts.findIndex((prompt) => prompt.name === payload.name);
  if (index >= 0) {
    state.prompts[index] = payload;
  }
  renderSelectedPrompt();
}

async function savePromptAs() {
  const name = cleanPromptName(els.promptName.value);
  if (!name) {
    throw new Error("Name the prompt before saving it.");
  }
  const payload = await api(`/api/prompts/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: els.promptText.value }),
  });
  const index = state.prompts.findIndex((prompt) => prompt.name === payload.name);
  if (index >= 0) {
    state.prompts[index] = payload;
  } else {
    state.prompts.push(payload);
    state.prompts.sort((left, right) => left.name.localeCompare(right.name, undefined, { numeric: true }));
  }
  await persistPromptChoice(payload.name);
  renderPrompts();
}

async function deleteSelectedPrompt() {
  const name = els.promptSelect.value;
  if (!name) return;
  if (!window.confirm(`Delete prompt "${name}"? Existing runs keep their saved prompt hash and results.`)) return;
  const payload = await api(`/api/prompts/${encodeURIComponent(name)}`, { method: "DELETE" });
  state.prompts = payload.prompts || [];
  applyTestState(payload.tests || {});
  renderPrompts();
  updateReadyState();
}

function cleanPromptName(value) {
  return String(value || "").trim().replace(/\.txt$/i, "");
}

async function persistPromptChoice(promptName) {
  if (!promptName || !state.activeTestId) return;
  const payload = await api("/api/tests/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ test_id: state.activeTestId, prompt: promptName }),
  });
  applyTestState(payload);
}

function filesFromInput(fileList) {
  return Array.from(fileList || []).filter((file) => /\.(jpe?g|png)$/i.test(file.name));
}

async function filesFromDrop(event) {
  const items = Array.from(event.dataTransfer.items || []);
  const entries = items
    .map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null))
    .filter(Boolean);
  if (!entries.length) {
    return filesFromInput(event.dataTransfer.files);
  }
  const files = [];
  for (const entry of entries) {
    await readEntry(entry, "", files);
  }
  return files.filter((file) => /\.(jpe?g|png)$/i.test(file.name));
}

function readEntry(entry, prefix, files) {
  return new Promise((resolve, reject) => {
    if (entry.isFile) {
      entry.file((file) => {
        Object.defineProperty(file, "relativePath", {
          value: `${prefix}${file.name}`,
          configurable: true,
        });
        files.push(file);
        resolve();
      }, reject);
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const directoryPrefix = `${prefix}${entry.name}/`;
      const readBatch = () => {
        reader.readEntries(async (entries) => {
          if (!entries.length) {
            resolve();
            return;
          }
          for (const child of entries) {
            await readEntry(child, directoryPrefix, files);
          }
          readBatch();
        }, reject);
      };
      readBatch();
    } else {
      resolve();
    }
  });
}

async function uploadFiles(files) {
  if (!files.length) {
    throw new Error("No JPEG or PNG images found in that folder.");
  }
  const form = new FormData();
  const firstPath = files[0].webkitRelativePath || files[0].relativePath || files[0].name;
  const folderLabel = firstPath.includes("/") ? firstPath.split("/")[0] : "browser-folder";
  form.append("folder_label", folderLabel);
  files.forEach((file) => {
    const relativePath = file.webkitRelativePath || file.relativePath || file.name;
    form.append("files", file, relativePath);
  });
  const session = await api("/api/sessions", { method: "POST", body: form });
  state.session = session;
  renderImages(session);
  applyTestState(await api("/api/tests"));
  updateReadyState();
}

function renderImages(session) {
  els.imageCount.textContent = `${session.images.length} image${session.images.length === 1 ? "" : "s"}`;
  els.imageList.classList.remove("empty");
  els.imageList.innerHTML = "";
  session.images.slice(0, 120).forEach((name) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `<span>${escapeHtml(name)}</span>`;
    els.imageList.append(row);
  });
  if (session.images.length > 120) {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `<span>${session.images.length - 120} more images...</span>`;
    els.imageList.append(row);
  }
}

function clearImages() {
  state.session = null;
  els.imageCount.textContent = "No folder selected";
  els.imageList.classList.add("empty");
  els.imageList.textContent = "No images loaded.";
  updateReadyState();
}

function nextTestNumber() {
  return state.tests.reduce((highest, test) => {
    const match = String(test.id || "").match(/^vision_test_(\d+)$/);
    return match ? Math.max(highest, Number(match[1])) : highest;
  }, 0) + 1;
}

function openNewTestPanel() {
  const number = nextTestNumber();
  els.newTestName.value = `Vision Test #${number}`;
  els.newTestPrompt.value = `classify_v${number}`;
  const active = state.activeTest || {};
  els.newTestDetail.textContent = active.image_count
    ? `Will reuse ${active.image_count} images and ${active.baseline_count || 0} baseline verdicts from ${active.name || "the current test"}.`
    : "Will reuse the current test setup. If no images are linked yet, choose the folder after creating the test.";
  els.newTestPanel.hidden = false;
  els.newTestName.focus();
}

function closeNewTestPanel() {
  els.newTestPanel.hidden = true;
}

async function createNewTest() {
  els.createNewTest.disabled = true;
  try {
    const payload = await api("/api/tests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: els.newTestName.value,
        prompt_name: els.newTestPrompt.value,
        source_test_id: state.activeTestId,
      }),
    });
    closeNewTestPanel();
    applyTestState(payload);
    state.prompts = payload.prompts || state.prompts;
    renderProfiles();
    renderPrompts();
    if (payload.session) {
      state.session = payload.session;
      renderImages(payload.session);
    } else {
      clearImages();
    }
    enableResultsTab();
    await loadResults();
    updateReadyState();
  } finally {
    els.createNewTest.disabled = false;
  }
}

async function switchTest(testId) {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  resetJobView();
  const payload = await api("/api/tests/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ test_id: testId }),
  });
  applyTestState(payload);
  renderProfiles();
  if (payload.session) {
    state.session = payload.session;
    renderImages(payload.session);
  } else {
    clearImages();
  }
  renderPrompts();
  enableResultsTab();
  await loadResults();
  await recoverJobLog();
  updateReadyState();
}

async function refreshModels() {
  els.modelList.className = "model-list empty";
  els.modelList.textContent = "Reading LM Studio models...";
  const payload = await api(`/api/models?base_url=${encodeURIComponent(els.baseUrl.value)}`);
  state.models = payload.models || [];
  state.selectedModels.clear();
  renderModels();
  updateReadyState();
}

function renderModels() {
  const showAll = els.showAllModels.checked;
  const models = state.models.filter((model) => showAll || model.vision);
  els.modelList.innerHTML = "";
  els.modelList.className = "model-list";
  if (!models.length) {
    els.modelList.className = "model-list empty";
    els.modelList.textContent = "No vision-capable models were detected. Toggle “Show non-vision or uncertain models” if LM Studio metadata is incomplete.";
    return;
  }
  models.forEach((model) => {
    const row = document.createElement("div");
    row.className = "model-row";
    const checked = state.selectedModels.has(model.id) ? "checked" : "";
    const visionLabel = model.vision ? "vision" : "uncertain";
    const loadedLabel = model.loaded ? "loaded" : "available";
    row.innerHTML = `
      <label>
        <input type="checkbox" value="${escapeHtml(model.id)}" ${checked} />
        <span>${escapeHtml(model.display_name || model.id)}</span>
      </label>
      <span class="model-meta">${escapeHtml(visionLabel)} · ${escapeHtml(loadedLabel)}</span>
    `;
    row.querySelector("input").addEventListener("change", (event) => {
      if (event.target.checked) {
        state.selectedModels.add(model.id);
      } else {
        state.selectedModels.delete(model.id);
      }
      updateReadyState();
    });
    els.modelList.append(row);
  });
}

async function startRun() {
  const payload = {
    test_id: state.activeTestId,
    session_id: state.session.id,
    prompt: els.promptSelect.value,
    profile: els.profileSelect.value,
    models: Array.from(state.selectedModels),
    base_url: els.baseUrl.value,
    timeout: 120,
  };
  const job = await api("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.activeJobId = job.id;
  rememberJob(job.id);
  els.logList.textContent = "Starting run...";
  document.querySelector('.tab[data-tab="run"]').disabled = false;
  showTab("run");
  pollJob();
  state.pollTimer = window.setInterval(pollJob, 1000);
}

async function pollJob() {
  if (!state.activeJobId) return;
  let job;
  try {
    job = await api(`/api/jobs/${encodeURIComponent(state.activeJobId)}`);
  } catch (error) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
    els.jobState.textContent = "Unavailable";
    els.logList.textContent = `The saved run log could not be recovered: ${error.message}`;
    return;
  }
  renderJob(job);
  if (job.status === "complete" || job.status === "failed") {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
    enableResultsTab();
    await loadResults();
    if (job.status === "complete") {
      showTab("results");
    }
  }
}

async function recoverJobLog() {
  const storedJobId = window.localStorage.getItem(jobStorageKey())
    || window.localStorage.getItem(JOB_STORAGE_KEY);
  if (storedJobId) {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(storedJobId)}`);
      if (!job.test_id || !state.activeTestId || job.test_id === state.activeTestId) {
        useRecoveredJob(job);
        return;
      }
    } catch (_) {
      window.localStorage.removeItem(jobStorageKey());
    }
  }

  try {
    const query = state.activeTestId ? `?test_id=${encodeURIComponent(state.activeTestId)}` : "";
    const payload = await api(`/api/jobs${query}`);
    if (payload.latest && payload.latest.id) {
      useRecoveredJob(payload.latest);
    }
  } catch (_) {
    // Older running servers may not expose the job list until the next restart.
  }
}

function useRecoveredJob(job) {
  state.activeJobId = job.id;
  rememberJob(job.id);
  document.querySelector('.tab[data-tab="run"]').disabled = false;
  renderJob(job);
  if (job.status === "queued" || job.status === "running") {
    if (!state.pollTimer) {
      state.pollTimer = window.setInterval(pollJob, 1000);
    }
    showTab("run");
  }
}

function resetJobView() {
  state.activeJobId = null;
  els.jobState.textContent = "Idle";
  els.progressBar.style.width = "0";
  els.currentModel.textContent = "Model: none";
  els.currentImage.textContent = "Image: none";
  els.logList.textContent = "No run started.";
}

function rememberJob(jobId) {
  if (jobId) {
    window.localStorage.setItem(jobStorageKey(), jobId);
  }
}

function jobStorageKey() {
  return `${JOB_STORAGE_KEY}:${state.activeTestId || "default"}`;
}

function renderJob(job) {
  els.jobState.textContent = job.status;
  const total = job.total_images || 0;
  const completed = job.completed_images || 0;
  const percent = total ? Math.min(100, Math.round((completed / total) * 100)) : 0;
  els.progressBar.style.width = `${percent}%`;
  els.currentModel.textContent = `Model: ${job.current_model || "none"}`;
  els.currentImage.textContent = `Image: ${job.current_image || "none"}`;
  const lines = (job.logs || []).map((entry) => `${entry.at}  ${entry.message}`);
  if (job.error) lines.push(`ERROR: ${job.error}`);
  els.logList.textContent = lines.join("\n") || "No log messages yet.";
  els.logList.scrollTop = els.logList.scrollHeight;
}

async function loadResults() {
  const payload = await api("/api/results");
  renderResults(payload);
  const report = await api("/api/report");
  els.reportPreview.textContent = report || "No report has been generated yet.";
}

function renderResults(payload) {
  const runs = payload.runs || [];
  state.resultsRuns = runs;
  els.resultsSummary.innerHTML = "";
  [
    ["Test", payload.test_name || "current"],
    ["Default profile", payload.default_profile || "VISION_HQ"],
    ["Runs", runs.length],
    ["Baseline items", payload.codex_verdict_count || 0],
    ["Dataset", payload.batch_path || "none"],
    ["Report", payload.report_exists ? "ready" : "missing"],
  ].forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "summary-item";
    item.innerHTML = `<strong>${escapeHtml(String(value))}</strong><span>${escapeHtml(label)}</span>`;
    els.resultsSummary.append(item);
  });
  renderResultsTable();
}

function renderResultsTable() {
  els.resultsTable.innerHTML = "";
  updateSortHeaders();
  sortedRuns().forEach((run) => {
    const stats = run.batch_stats || {};
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(run.run_id || "")}</td>
      <td>${escapeHtml(formatTimestamp(run.run_timestamp))}</td>
      <td>${escapeHtml(formatDuration(stats.total_time_sec))}</td>
      <td>${escapeHtml(run.model || "")}</td>
      <td>${escapeHtml(run.prompt || "")}</td>
      <td>${escapeHtml(run.profile || "VISION_HQ")}</td>
      <td>${stats.total_images || 0}</td>
      <td>${formatNumber(stats.avg_tokens_per_sec)}</td>
      <td>${formatPercent(stats.codex_agreement_rate)}</td>
      <td>${stats.error_count || 0}</td>
      <td></td>
    `;
    const actionCell = row.querySelector("td:last-child");
    const deleteButton = document.createElement("button");
    deleteButton.className = "danger small";
    deleteButton.type = "button";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", () => deleteRun(run.run_id));
    actionCell.append(deleteButton);
    els.resultsTable.append(row);
  });
}

function sortedRuns() {
  const direction = state.resultsSort.direction === "desc" ? -1 : 1;
  return [...state.resultsRuns].sort((left, right) => {
    const leftValue = sortValue(left, state.resultsSort.key);
    const rightValue = sortValue(right, state.resultsSort.key);
    if (typeof leftValue === "number" && typeof rightValue === "number") {
      return (leftValue - rightValue) * direction;
    }
    return String(leftValue).localeCompare(String(rightValue), undefined, {
      numeric: true,
      sensitivity: "base",
    }) * direction;
  });
}

function sortValue(run, key) {
  const stats = run.batch_stats || {};
  if (key === "run_timestamp") return Date.parse(run.run_timestamp || "") || 0;
  if (key === "total_time_sec") return Number(stats.total_time_sec || 0);
  if (key === "total_images") return Number(stats.total_images || 0);
  if (key === "avg_tokens_per_sec") return Number(stats.avg_tokens_per_sec || 0);
  if (key === "codex_agreement_rate") return Number(stats.codex_agreement_rate ?? -1);
  if (key === "error_count") return Number(stats.error_count || 0);
  return run[key] || "";
}

function updateSortHeaders() {
  document.querySelectorAll(".sort-header").forEach((button) => {
    const active = button.dataset.sort === state.resultsSort.key;
    button.classList.toggle("active", active);
    button.dataset.direction = active ? state.resultsSort.direction : "";
    button.setAttribute(
      "aria-sort",
      active ? (state.resultsSort.direction === "asc" ? "ascending" : "descending") : "none",
    );
  });
}

function sortResultsBy(key) {
  if (state.resultsSort.key === key) {
    state.resultsSort.direction = state.resultsSort.direction === "asc" ? "desc" : "asc";
  } else {
    state.resultsSort.key = key;
    state.resultsSort.direction = "desc";
  }
  renderResultsTable();
}

async function clearCurrentResults() {
  const runCount = state.resultsRuns.length;
  if (!runCount) return;
  if (!window.confirm(`Clear all ${runCount} run result${runCount === 1 ? "" : "s"} for this test? The image set and baseline verdicts will stay.`)) return;
  const payload = await api("/api/tests/results", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ test_id: state.activeTestId }),
  });
  applyTestState(payload.tests || {});
  renderResults(payload.results || { runs: [] });
  els.reportPreview.textContent = payload.report || "No report has been generated yet.";
}

async function deleteRun(runId) {
  if (!runId) return;
  if (!window.confirm(`Delete ${runId}? This removes that run from the current test report.`)) return;
  const path = `/api/runs/${encodeURIComponent(runId)}?test_id=${encodeURIComponent(state.activeTestId || "")}`;
  const payload = await api(path, { method: "DELETE" });
  applyTestState(payload.tests || {});
  renderResults(payload.results || { runs: [] });
  els.reportPreview.textContent = payload.report || "No report has been generated yet.";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatNumber(value) {
  const number = Number(value || 0);
  return number.toFixed(1);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatTimestamp(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(value) {
  const totalSeconds = Math.round(Number(value || 0));
  if (!totalSeconds) return "0s";
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (!minutes) return `${seconds}s`;
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => showTab(button.dataset.tab));
});

document.querySelectorAll(".sort-header").forEach((button) => {
  button.addEventListener("click", () => sortResultsBy(button.dataset.sort));
});

els.testSelect.addEventListener("change", () => switchTest(els.testSelect.value).catch((error) => alert(error.message)));
els.newTest.addEventListener("click", openNewTestPanel);
els.cancelNewTest.addEventListener("click", closeNewTestPanel);
els.createNewTest.addEventListener("click", () => createNewTest().catch((error) => alert(error.message)));
els.newTestPanel.addEventListener("click", (event) => {
  if (event.target === els.newTestPanel) closeNewTestPanel();
});

els.folderInput.addEventListener("change", (event) => {
  uploadFiles(filesFromInput(event.target.files)).catch((error) => alert(error.message));
});

["dragenter", "dragover"].forEach((eventName) => {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragging");
  });
});

els.dropZone.addEventListener("drop", async (event) => {
  try {
    const files = await filesFromDrop(event);
    await uploadFiles(files);
  } catch (error) {
    alert(error.message);
  }
});

els.refreshModels.addEventListener("click", () => refreshModels().catch((error) => {
  els.modelList.className = "model-list empty";
  els.modelList.textContent = error.message;
}));

els.saveToken.addEventListener("click", () => saveToken().catch((error) => {
  els.tokenStatus.textContent = error.message;
}));
els.showAllModels.addEventListener("change", renderModels);
els.profileSelect.addEventListener("change", () => {
  renderSelectedProfile();
  persistProfile().catch((error) => alert(error.message));
});
els.promptSelect.addEventListener("change", () => {
  renderSelectedPrompt();
  persistPromptChoice(els.promptSelect.value).catch((error) => alert(error.message));
});
els.editPrompt.addEventListener("click", enablePromptEditing);
els.savePrompt.addEventListener("click", () => savePrompt().catch((error) => alert(error.message)));
els.savePromptAs.addEventListener("click", () => savePromptAs().catch((error) => alert(error.message)));
els.deletePrompt.addEventListener("click", () => deleteSelectedPrompt().catch((error) => alert(error.message)));
els.startRun.addEventListener("click", () => startRun().catch((error) => alert(error.message)));
els.reloadResults.addEventListener("click", () => loadResults().catch((error) => alert(error.message)));
els.clearResults.addEventListener("click", () => clearCurrentResults().catch((error) => alert(error.message)));

loadConfig().catch((error) => {
  setMessage("Startup failed.", error.message);
});
