const form = document.querySelector("#analysis-form");
const urlInput = document.querySelector("#url-input");
const modelSelect = document.querySelector("#model-select");
const modelHelp = document.querySelector("#model-help");
const analyzeButton = document.querySelector("#analyze-button");
const formError = document.querySelector("#form-error");
const report = document.querySelector("#report");
const resultTitle = document.querySelector("#result-title");
const resultStatus = document.querySelector("#result-status");
const resultSummary = document.querySelector("#result-summary");
const resultContent = document.querySelector("#result-content");
const modelInventory = new Map();

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function setError(message) {
  formError.textContent = message;
  formError.hidden = !message;
  urlInput.setAttribute("aria-invalid", message ? "true" : "false");
}

function updateModelControls() {
  const info = modelInventory.get(modelSelect.value);
  modelHelp.textContent = info ? info.input_scope : "Select an analysis mode.";
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error();
    const data = await response.json();
    modelSelect.replaceChildren();
    data.models.forEach((info) => {
      modelInventory.set(info.model_id, info);
      const option = element("option", info.display_name);
      option.value = info.model_id;
      modelSelect.append(option);
    });
    modelSelect.value = "automatic";
    updateModelControls();
  } catch {
    modelHelp.textContent = "Model inventory unavailable; automatic mode remains selected.";
  }
}

function metric(label, value) {
  const node = element("section", null, "mini-metric");
  node.append(element("span", label), element("strong", value));
  return node;
}

function scoreCell(score) {
  const probability = score.phishing_probability;
  if (probability === null) return element("td", "Unavailable");
  const percent = Math.round(probability * 100);
  const cell = element("td", null, "score-cell");
  const bar = element("span", null, "score-bar");
  const fill = element("span");
  fill.style.width = `${percent}%`;
  bar.append(fill);
  cell.append(bar, element("span", `${percent}%`, "score-value"));
  return cell;
}

function renderScores(scores) {
  const section = element("article", null, "evidence-section");
  const heading = element("header", null, "section-heading");
  heading.append(element("h3", "Model comparison"), element("span", `${scores.length} participating`));
  section.append(heading);
  const wrap = element("figure", null, "table-wrap");
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headingRow = document.createElement("tr");
  ["Model", "Verdict", "Phishing score"].forEach((label) => headingRow.append(element("th", label)));
  head.append(headingRow);
  const body = document.createElement("tbody");
  scores.forEach((score) => {
    const row = document.createElement("tr");
    row.append(element("td", score.display_name), element("td", score.predicted_label || score.status, "model-verdict"), scoreCell(score));
    body.append(row);
  });
  table.append(head, body);
  wrap.append(table);
  section.append(wrap);
  return section;
}

function renderExplanation(data) {
  const stack = element("aside", null, "explanation-stack");
  const signals = element("section", null, "explanation-card");
  signals.append(element("h3", "Active URL patterns"));
  if (data.signals.length) {
    const list = element("ul", null, "signal-list");
    data.signals.slice(0, 4).forEach((signal) => {
      const item = document.createElement("li");
      item.append(element("code", signal.ngram), element("span", signal.direction));
      list.append(item);
    });
    signals.append(list);
  } else {
    signals.append(element("p", "This model does not expose URL-text pattern contributions."));
  }
  stack.append(signals);

  if (data.warnings.length) {
    const warnings = element("section", null, "explanation-card warning-card");
    warnings.append(element("h3", "Why caution remains"));
    const list = element("ul", null, "warning-list");
    data.warnings.slice(0, 3).forEach((warning) => list.append(element("li", warning)));
    warnings.append(list);
    stack.append(warnings);
  }
  return stack;
}

function renderResult(data) {
  const label = data.predicted_label;
  const score = Math.round(data.phishing_probability * 100);
  const agreement = Math.round(data.agreement * 100);
  const scanLabel = data.deep_scan_status === "complete" ? "Complete" : data.deep_scan_status === "not_requested" ? "URL only" : "Unavailable";

  report.hidden = false;
  resultTitle.textContent = label === "uncertain" ? "Evidence conflicts" : label === "phishing" ? "Potential phishing detected" : "Likely legitimate";
  resultStatus.className = `result-status ${label}`;
  resultStatus.textContent = label;
  resultSummary.textContent = label === "uncertain" ? "Models disagree or the combined score is too close to call safely." : label === "phishing" ? "Evidence leans toward phishing. Do not enter credentials until independently verified." : "Evidence leans legitimate, but this result cannot guarantee safety.";

  resultContent.replaceChildren();
  const overview = element("section", null, "verdict-overview");
  const metrics = element("section", null, "metric-grid");
  metrics.append(metric("Phishing score", `${score}%`), metric("Model agreement", `${agreement}%`), metric("Live evidence", scanLabel));
  const scale = element("figure", null, "risk-scale");
  const meter = element("p", null, "meter");
  const fill = element("span");
  fill.style.width = `${score}%`;
  meter.append(fill);
  const labels = element("figcaption", null, "risk-labels");
  labels.append(element("span", "Leans legitimate"), element("span", "Review"), element("span", "Leans phishing"));
  scale.append(meter, labels);
  overview.append(metrics, scale);

  const evidence = element("section", null, "evidence-grid");
  evidence.append(renderScores(data.model_scores), renderExplanation(data));
  resultContent.append(overview, evidence, element("p", `${data.probability_note} ${data.analysis_scope}`, "result-note"));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  report.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

async function analyzeUrl(event) {
  event.preventDefault();
  const url = urlInput.value.trim();
  if (!url) { setError("Enter a URL to analyse."); urlInput.focus(); return; }
  setError("");
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analysing…";
  try {
    const deepScan = !["tfidf", "minilm"].includes(modelSelect.value);
    const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url, model: modelSelect.value, deep_scan: deepScan }) });
    const data = await response.json();
    if (!response.ok) throw new Error(response.status === 422 ? "Enter a valid public HTTP or HTTPS URL." : data.detail || "Analysis is unavailable.");
    renderResult(data);
  } catch (error) {
    setError(error.message || "Analysis failed. Try again.");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyse URL";
  }
}

document.querySelectorAll(".samples button[data-url]").forEach((button) => button.addEventListener("click", () => {
  urlInput.value = button.dataset.url || "";
  modelSelect.value = button.dataset.model || "automatic";
  updateModelControls();
  setError("");
  urlInput.focus();
}));
modelSelect.addEventListener("change", updateModelControls);
form.addEventListener("submit", analyzeUrl);
loadModels();
