const STORAGE_KEY = "recruiter_active_job_session";

const messageList = document.getElementById("message-list");
const jobPanelBody = document.getElementById("job-panel-body");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const errorSlot = document.getElementById("error-banner-slot");
const startNewBtn = document.getElementById("start-new-btn");
const attachBtn = document.getElementById("attach-btn");
const fileInput = document.getElementById("file-input");
const attachmentChipSlot = document.getElementById("attachment-chip-slot");
const savedIndicatorEl = document.getElementById("saved-indicator");
const jobDrawer = document.getElementById("job-panel");
const jobDrawerOverlay = document.getElementById("job-drawer-overlay");
const jobDetailsToggle = document.getElementById("job-details-toggle");
const jobDetailsClose = document.getElementById("job-details-close");
const jobDetailsDot = document.getElementById("job-details-dot");

let sessionId = null;
let currentPhase = null;
let pendingFile = null;
let lastSavedAt = null;

const FIELD_LABELS = {
  job_title: "Job Title",
  job_category: "Job Category",
  experience: "Experience",
  location: "Location",
  work_mode: "Work Mode",
  employment_type: "Employment Type",
  education: "Education",
  salary: "Salary",
  additional_information: "Additional Info",
};

function getStoredSession() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
  } catch (_) {
    return null;
  }
}

function setStoredSession(id, jobTitle) {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ session_id: id, job_title: jobTitle || null, updated_at: new Date().toISOString() })
  );
}

function clearStoredSession() {
  localStorage.removeItem(STORAGE_KEY);
}

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function avatarEl() {
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "A";
  return avatar;
}

function appendMessage(role, content) {
  const row = document.createElement("div");
  row.className = `message-row ${role === "user" ? "user" : "ai"}`;
  if (role !== "user") row.appendChild(avatarEl());
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  row.appendChild(bubble);
  messageList.appendChild(row);
  messageList.scrollTop = messageList.scrollHeight;
  return row;
}

// Shows the real, backend-driven processing label for the turn in progress (see
// stream_graph_turn / ROUTE_STATUS_LABELS in chat.py) — never a generic "..." indicator, and
// never a fabricated label the backend didn't actually emit. Called once per SSE status event,
// so the same row's text updates live as the graph moves through real steps.
function showProcessingStatus(label) {
  let row = document.getElementById("processing-status");
  if (!row) {
    row = document.createElement("div");
    row.className = "message-row ai processing-status";
    row.id = "processing-status";
    row.appendChild(avatarEl());
    const bubble = document.createElement("div");
    bubble.className = "bubble processing-bubble";
    const dot = document.createElement("span");
    dot.className = "processing-dot";
    const text = document.createElement("span");
    text.className = "processing-text";
    bubble.appendChild(dot);
    bubble.appendChild(text);
    row.appendChild(bubble);
    messageList.appendChild(row);
  }
  row.querySelector(".processing-text").textContent = label;
  messageList.scrollTop = messageList.scrollHeight;
}

function hideProcessingStatus() {
  const el = document.getElementById("processing-status");
  if (el) el.remove();
}

function showError(message) {
  clearChildren(errorSlot);
  const banner = document.createElement("div");
  banner.className = "error-banner";
  banner.textContent = message;
  errorSlot.appendChild(banner);
}

function clearError() {
  clearChildren(errorSlot);
}

// Full rebuild from the authoritative history rather than an incremental append — the backend
// recomputes is_selected for every jd_document message against the current selected_version each
// turn (see _to_chat_messages in chat.py), so this is what keeps an already-rendered JD card's
// "Selected" tag and Choose button in sync after a later selection turn, not just new messages.
function renderMessages(messages, askingAboutField) {
  clearChildren(messageList);
  if (!messages || messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent =
      'Tell me about the role you’re hiring for — title, experience, location, and any must-have skills.';
    messageList.appendChild(empty);
    return;
  }
  let lastRow = null;
  let lastWasAssistantText = false;
  messages.forEach((m) => {
    if (m.jd_document) {
      lastRow = appendJdMessage(m.jd_version, m.jd_document, m.is_selected);
      lastWasAssistantText = false;
    } else {
      lastRow = appendMessage(m.role, m.content);
      lastWasAssistantText = m.role !== "user";
    }
  });
  // The backend only ever sets asking_about_field when its latest reply is a live question
  // about one specific optional field (see apply_updates in nodes.py) — never fabricated
  // client-side, and never shown once the conversation has moved past that question.
  if (askingAboutField && lastWasAssistantText && lastRow) {
    appendSkipButton(lastRow, askingAboutField);
  }
}

function appendSkipButton(afterRow, field) {
  const row = document.createElement("div");
  row.className = "skip-row";
  const skipBtn = document.createElement("button");
  skipBtn.type = "button";
  skipBtn.className = "btn btn-small skip-btn";
  skipBtn.textContent = "Skip this";
  skipBtn.addEventListener("click", () => {
    sendMessage("I don't have that information for this role — let's skip it.");
  });
  row.appendChild(skipBtn);
  afterRow.insertAdjacentElement("afterend", row);
  messageList.scrollTop = messageList.scrollHeight;
}

function appendJdMessage(versionKey, jd, isSelected) {
  const row = document.createElement("div");
  row.className = "message-row ai jd-message-row";
  row.appendChild(avatarEl());
  const card = jdCard(versionKey, jd, isSelected);
  if (!isSelected) {
    const footer = document.createElement("div");
    footer.className = "jd-card-footer";
    const chooseBtn = document.createElement("button");
    chooseBtn.type = "button";
    chooseBtn.className = "btn btn-primary btn-small";
    chooseBtn.textContent = `Choose Version ${versionKey}`;
    chooseBtn.addEventListener("click", () => sendMessage(`I choose Version ${versionKey}.`));
    footer.appendChild(chooseBtn);
    card.appendChild(footer);
  }
  row.appendChild(card);
  messageList.appendChild(row);
  messageList.scrollTop = messageList.scrollHeight;
  return row;
}

function fieldRow(label, value, isEmpty) {
  const row = document.createElement("div");
  row.className = "field-row";
  const labelEl = document.createElement("div");
  labelEl.className = "field-label";
  labelEl.textContent = label;
  const valueEl = document.createElement("div");
  valueEl.className = "field-value" + (isEmpty ? " empty" : "");
  valueEl.textContent = isEmpty ? "Not specified" : value;
  row.appendChild(labelEl);
  row.appendChild(valueEl);
  return row;
}

function chipListRow(label, items) {
  const row = document.createElement("div");
  row.className = "field-row";
  const labelEl = document.createElement("div");
  labelEl.className = "field-label";
  labelEl.textContent = label;
  const valueEl = document.createElement("div");
  valueEl.className = "field-value" + (items.length === 0 ? " empty" : "");
  if (items.length === 0) {
    valueEl.textContent = "Not specified";
  } else {
    const chipList = document.createElement("div");
    chipList.className = "chip-list";
    items.forEach((item) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = item;
      chipList.appendChild(chip);
    });
    valueEl.appendChild(chipList);
  }
  row.appendChild(labelEl);
  row.appendChild(valueEl);
  return row;
}

function groupHeading(text) {
  const h3 = document.createElement("h3");
  h3.textContent = text;
  return h3;
}

function renderAlreadyPostedBanner(data, record) {
  const box = document.createElement("div");
  box.className = "publish-success";

  const check = document.createElement("div");
  check.className = "check";
  check.textContent = "✓ Job Already Posted";
  box.appendChild(check);

  const jobId = document.createElement("div");
  jobId.className = "job-id-big";
  jobId.textContent = record.job_id;
  box.appendChild(jobId);

  const titleLine = document.createElement("div");
  titleLine.className = "job-title-line";
  titleLine.textContent = data.job_state.job_title || "";
  box.appendChild(titleLine);

  const metaLine = document.createElement("div");
  metaLine.className = "job-meta-line";
  metaLine.textContent = [data.job_state.location, data.job_state.work_mode].filter(Boolean).join(" · ");
  box.appendChild(metaLine);

  const actions = document.createElement("div");
  actions.className = "publish-success-actions";
  const viewLink = document.createElement("a");
  viewLink.className = "btn btn-primary";
  viewLink.textContent = "View Published Job";
  viewLink.href = `/jobs.html?job_id=${encodeURIComponent(record.job_id)}`;
  actions.appendChild(viewLink);
  box.appendChild(actions);

  const hint = document.createElement("p");
  hint.className = "already-posted-hint";
  hint.textContent = "Want to change something? Just tell me what to update — it won't go live until you confirm.";
  box.appendChild(hint);

  return box;
}

function renderEditingBanner(data) {
  const box = document.createElement("div");
  box.className = "editing-banner";

  const title = document.createElement("div");
  title.className = "editing-banner-title";
  title.textContent = `Editing ${data.job_record.job_id} — unpublished changes`;
  box.appendChild(title);

  const hint = document.createElement("p");
  hint.className = "editing-banner-hint";
  const canPublish = Boolean(data.selected_version) && !data.jd_stale;
  hint.textContent = canPublish
    ? "Your changes are staged. Publish when ready to make them live."
    : data.jd_stale
      ? "The job description needs to be regenerated or refined to match your changes before publishing."
      : "Select a job description version before publishing this edit.";
  box.appendChild(hint);

  const publishBtn = document.createElement("button");
  publishBtn.type = "button";
  publishBtn.className = "btn btn-primary";
  publishBtn.textContent = "Publish Edit";
  publishBtn.disabled = !canPublish;
  publishBtn.addEventListener("click", () => sendMessage("Please publish this edit."));
  box.appendChild(publishBtn);

  return box;
}

function updateJobDetailsAttention(data) {
  const needsAttention = data.phase === "publish_confirm" || data.phase === "editing";
  jobDetailsDot.hidden = !needsAttention;
}

function renderJobPanel(data) {
  clearChildren(jobPanelBody);
  updateJobDetailsAttention(data);
  const jobState = data.job_state || {};
  const record = data.job_record;

  if (record && record.status === "published" && data.phase === "published") {
    jobPanelBody.appendChild(renderAlreadyPostedBanner(data, record));
  } else if (record && record.status === "published" && data.phase === "editing") {
    jobPanelBody.appendChild(renderEditingBanner(data));
  }

  // Status / draft indicator
  const statusRow = document.createElement("div");
  statusRow.className = "status-row";

  const badge = document.createElement("span");
  if (record && record.status === "published" && data.phase === "editing") {
    badge.className = "badge badge-draft";
    const dot = document.createElement("span");
    dot.className = "badge-dot";
    badge.appendChild(dot);
    badge.appendChild(document.createTextNode("Editing — unpublished changes"));
  } else if (record && record.status === "published") {
    badge.className = "badge badge-published";
    const dot = document.createElement("span");
    dot.className = "badge-dot";
    badge.appendChild(dot);
    badge.appendChild(document.createTextNode(`Published${record.job_id ? " — " + record.job_id : ""}`));
  } else if (record) {
    badge.className = "badge badge-draft";
    const dot = document.createElement("span");
    dot.className = "badge-dot";
    badge.appendChild(dot);
    badge.appendChild(document.createTextNode("Draft — saved automatically"));
  }
  if (record) statusRow.appendChild(badge);

  const phaseBadge = document.createElement("span");
  phaseBadge.className = "badge";
  phaseBadge.style.background = "#eef4f2";
  phaseBadge.style.color = "#2f5d50";
  phaseBadge.textContent = `Phase: ${data.phase}`;
  statusRow.appendChild(phaseBadge);

  jobPanelBody.appendChild(statusRow);

  // Completeness bar (UX indicator only)
  const completenessWrap = document.createElement("div");
  completenessWrap.className = "completeness-wrap";
  const label = document.createElement("div");
  label.className = "completeness-label";
  const labelLeft = document.createElement("span");
  labelLeft.textContent = "Job completeness";
  const labelRight = document.createElement("span");
  labelRight.textContent = `${data.completeness_pct}%`;
  label.appendChild(labelLeft);
  label.appendChild(labelRight);
  const bar = document.createElement("div");
  bar.className = "completeness-bar";
  const fill = document.createElement("div");
  fill.className = "completeness-fill";
  fill.style.width = `${data.completeness_pct}%`;
  bar.appendChild(fill);
  completenessWrap.appendChild(label);
  completenessWrap.appendChild(bar);
  jobPanelBody.appendChild(completenessWrap);

  // Missing essential note
  if (data.missing_essential && data.missing_essential.length > 0) {
    const note = document.createElement("div");
    note.className = "missing-note";
    note.textContent = `Still needed: ${data.missing_essential.join(", ")}`;
    jobPanelBody.appendChild(note);
  }

  // Stale JD note — job details changed after the description was generated
  if (data.jd_stale && data.jd_versions) {
    const stale = document.createElement("div");
    stale.className = "stale-banner";
    stale.textContent =
      "The job description is out of date — job details changed since it was generated. Ask me to regenerate or refine it.";
    jobPanelBody.appendChild(stale);
  }

  // Job basics
  const basics = document.createElement("div");
  basics.className = "field-group";
  basics.appendChild(groupHeading("Job Basics"));
  basics.appendChild(fieldRow("Job Title", jobState.job_title, !jobState.job_title));
  basics.appendChild(fieldRow("Job Category", jobState.job_category, !jobState.job_category));
  basics.appendChild(fieldRow("Experience", jobState.experience, !jobState.experience));
  basics.appendChild(fieldRow("Location", jobState.location, !jobState.location));
  basics.appendChild(fieldRow("Work Mode", jobState.work_mode, !jobState.work_mode));
  basics.appendChild(fieldRow("Employment Type", jobState.employment_type, !jobState.employment_type));
  jobPanelBody.appendChild(basics);

  // Requirements
  const requirements = document.createElement("div");
  requirements.className = "field-group";
  requirements.appendChild(groupHeading("Requirements"));
  requirements.appendChild(chipListRow("Required Skills", jobState.required_skills || []));
  requirements.appendChild(chipListRow("Preferred Skills", jobState.preferred_skills || []));
  requirements.appendChild(chipListRow("Responsibilities", jobState.responsibilities || []));
  requirements.appendChild(fieldRow("Education", jobState.education, !jobState.education));
  jobPanelBody.appendChild(requirements);

  // Compensation & extra
  const extra = document.createElement("div");
  extra.className = "field-group";
  extra.appendChild(groupHeading("Compensation & Notes"));
  extra.appendChild(fieldRow("Salary", jobState.salary, !jobState.salary));
  extra.appendChild(fieldRow("Additional Info", jobState.additional_information, !jobState.additional_information));
  jobPanelBody.appendChild(extra);

  // Company context overrides (job-specific only — never the stored company profile)
  const overrides = jobState.company_overrides || {};
  const overrideKeys = Object.keys(overrides);
  if (overrideKeys.length > 0) {
    const overridesGroup = document.createElement("div");
    overridesGroup.className = "field-group";
    overridesGroup.appendChild(groupHeading("Company Context Overrides (this job only)"));
    overrideKeys.forEach((key) => {
      overridesGroup.appendChild(fieldRow(key.replace(/_/g, " "), overrides[key], false));
    });
    jobPanelBody.appendChild(overridesGroup);
  }

  renderJdStatusLine(data);
}

const JD_TEXT_FIELDS = [
  ["job_summary", "Job Summary"],
  ["about_role", "About the Role"],
  ["company_overview", "Company Overview"],
  ["why_company", "Why Join"],
];

const JD_LIST_FIELDS = [
  ["major_accountabilities", "Major Accountabilities"],
  ["minimum_requirements", "Minimum Requirements"],
  ["required_qualifications", "Required Qualifications"],
  ["preferred_qualifications", "Preferred Qualifications"],
  ["required_skills", "Required Skills"],
  ["stand_out", "Ways to Stand Out"],
  ["benefits", "Benefits / Employee Experience"],
];

function jdMetaLine(jd) {
  const parts = [jd.job_category, jd.employment_type, jd.location, jd.work_mode].filter(Boolean);
  return parts.join(" · ");
}

function jdCard(versionKey, jd, isSelected) {
  const card = document.createElement("div");
  card.className = "jd-card" + (isSelected ? " selected" : "");

  const header = document.createElement("div");
  header.className = "jd-card-header";
  const title = document.createElement("span");
  title.textContent = `Version ${versionKey}`;
  header.appendChild(title);
  if (isSelected) {
    const tag = document.createElement("span");
    tag.className = "selected-tag";
    tag.textContent = "Selected";
    header.appendChild(tag);
  }
  card.appendChild(header);

  const body = document.createElement("div");
  body.className = "jd-card-body";

  const meta = jdMetaLine(jd);
  if (meta) {
    const metaField = document.createElement("div");
    metaField.className = "jd-field";
    const p = document.createElement("p");
    p.textContent = meta;
    metaField.appendChild(p);
    body.appendChild(metaField);
  }

  JD_TEXT_FIELDS.forEach(([key, label]) => {
    const value = jd[key];
    if (!value) return;
    const field = document.createElement("div");
    field.className = "jd-field";
    const h4 = document.createElement("h4");
    h4.textContent = label;
    const p = document.createElement("p");
    p.textContent = value;
    field.appendChild(h4);
    field.appendChild(p);
    body.appendChild(field);
  });

  JD_LIST_FIELDS.forEach(([key, label]) => {
    const items = jd[key];
    if (!items || items.length === 0) return;
    const field = document.createElement("div");
    field.className = "jd-field";
    const h4 = document.createElement("h4");
    h4.textContent = label;
    const ul = document.createElement("ul");
    items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      ul.appendChild(li);
    });
    field.appendChild(h4);
    field.appendChild(ul);
    body.appendChild(field);
  });

  card.appendChild(body);
  return card;
}

function renderJdStatusLine(data) {
  const jdVersions = data.jd_versions;
  const has1 = jdVersions && jdVersions["1"];
  const has2 = jdVersions && jdVersions["2"];
  if (!has1 && !has2) return;

  const heading = document.createElement("div");
  heading.className = "jd-section-title";
  heading.textContent = "Job Description";
  jobPanelBody.appendChild(heading);

  const wrap = document.createElement("div");
  wrap.className = "jd-status-line";
  wrap.textContent = data.selected_version
    ? `Selected: Version ${data.selected_version}`
    : "Two versions generated — see conversation";
  jobPanelBody.appendChild(wrap);
}

function updateSavedIndicator() {
  if (!savedIndicatorEl) return;
  savedIndicatorEl.textContent = lastSavedAt ? `Saved ${formatRelative(lastSavedAt)}` : "";
}

setInterval(updateSavedIndicator, 20000);

function renderAttachmentChip() {
  clearChildren(attachmentChipSlot);
  if (!pendingFile) return;
  const chip = document.createElement("div");
  chip.className = "attachment-chip";
  const name = document.createElement("span");
  name.textContent = `📎 ${pendingFile.name}`;
  chip.appendChild(name);
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "attachment-chip-remove";
  remove.textContent = "×";
  remove.setAttribute("aria-label", "Remove attachment");
  remove.addEventListener("click", () => {
    pendingFile = null;
    fileInput.value = "";
    renderAttachmentChip();
  });
  chip.appendChild(remove);
  attachmentChipSlot.appendChild(chip);
}

attachBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  pendingFile = file;
  renderAttachmentChip();
});

async function sendMessage(text) {
  clearError();
  const file = pendingFile;
  const displayText = file ? (text ? `📎 ${file.name}\n${text}` : `📎 ${file.name}`) : text;
  appendMessage("user", displayText);
  chatInput.value = "";
  chatInput.style.height = "auto";
  pendingFile = null;
  fileInput.value = "";
  renderAttachmentChip();
  sendBtn.disabled = true;
  showProcessingStatus(file ? "Reading the uploaded document..." : "Understanding your request...");

  try {
    const data = file
      ? await api.postChatUpload(sessionId, text, file, showProcessingStatus)
      : await api.postChat(sessionId, text, showProcessingStatus);
    hideProcessingStatus();
    sessionId = data.session_id;
    if (data.phase === "published") {
      // Settled — either a fresh publish or a completed edit. Nothing pending, safe to forget
      // so a plain reload starts a new job by default. (An in-progress edit has phase
      // "editing" here instead, so it's correctly kept resumable.)
      clearStoredSession();
      if (currentPhase && currentPhase !== "published") {
        const jobId = data.job_record && data.job_record.job_id;
        showToast(jobId ? `Job ${jobId} published successfully.` : "Job published successfully.", "success");
      }
    } else {
      setStoredSession(sessionId, data.job_state && data.job_state.job_title);
    }
    currentPhase = data.phase;
    renderMessages(data.messages, data.asking_about_field);
    lastSavedAt = (data.job_record && data.job_record.updated_at) || new Date().toISOString();
    updateSavedIndicator();
    renderJobPanel(data);
  } catch (err) {
    hideProcessingStatus();
    showError(err.message || "Something went wrong. Please try again.");
  } finally {
    sendBtn.disabled = false;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text && !pendingFile) return;
  sendMessage(text);
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 140)}px`;
});

startNewBtn.addEventListener("click", () => {
  clearStoredSession();
  sessionId = null;
  currentPhase = null;
  pendingFile = null;
  fileInput.value = "";
  renderAttachmentChip();
  lastSavedAt = null;
  updateSavedIndicator();
  clearError();
  renderMessages([]);
  renderJobPanel({ job_state: {}, job_record: null, phase: "collecting", completeness_pct: 0, missing_essential: [] });
  chatInput.focus();
});

function openJobDrawer() {
  // Drop the keyboard before opening — Job Details is reached from the header now, not the
  // composer, so there's no reason to keep the keyboard up (and leaving it up is exactly what
  // was making fixed-position elements misbehave on mobile while it was open).
  chatInput.blur();
  jobDrawer.classList.add("open");
  jobDrawerOverlay.classList.add("open");
}

function closeJobDrawer() {
  jobDrawer.classList.remove("open");
  jobDrawerOverlay.classList.remove("open");
}

// CSS dvh alone doesn't reliably shrink for the on-screen keyboard (Safari never does; Android
// Chrome varies by version), which is what let the drawer/keyboard visually collide. Mirroring
// the real visual viewport height into a custom property is what actually keeps the composer
// pinned directly above the keyboard on every mobile browser.
function syncAppHeight() {
  if (!window.visualViewport) return;
  document.documentElement.style.setProperty("--app-vh", `${window.visualViewport.height}px`);
}
if (window.visualViewport) {
  syncAppHeight();
  window.visualViewport.addEventListener("resize", syncAppHeight);
  window.visualViewport.addEventListener("scroll", syncAppHeight);
}

jobDetailsToggle.addEventListener("click", openJobDrawer);
jobDetailsClose.addEventListener("click", closeJobDrawer);
jobDrawerOverlay.addEventListener("click", closeJobDrawer);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && jobDrawer.classList.contains("open")) closeJobDrawer();
});

const logoutBtn = document.getElementById("logout-btn");
logoutBtn.addEventListener("click", async () => {
  try {
    await api.logout();
  } catch (_) {
    // Ignore — redirecting to /auth.html either way makes the client-side state moot.
  }
  window.location.href = "/auth.html";
});

async function init() {
  // Backend authorization is the real boundary (every /api/chat call re-checks session
  // ownership) — this is purely so an unauthenticated visitor isn't shown the chat shell first.
  try {
    await api.getMe();
  } catch (err) {
    window.location.href = "/auth.html";
    return;
  }

  renderJobPanel({ job_state: {}, job_record: null, phase: "collecting", completeness_pct: 0, missing_essential: [] });
  renderMessages([]);

  // Explicit navigation (e.g. "Continue" / "Edit Job" from the dashboard or admin) always wins
  // over the "last active" localStorage convenience, and loads regardless of published status —
  // that's the whole point of the edit-published-job flow.
  const params = new URLSearchParams(window.location.search);
  const explicitSessionId = params.get("session_id");

  const targetSessionId = explicitSessionId || (getStoredSession() || {}).session_id;
  if (!targetSessionId) return;

  sessionId = targetSessionId;
  try {
    const data = await api.getChat(sessionId);
    if (!explicitSessionId && data.job_record && data.job_record.status === "published" && data.phase === "published") {
      // Reached only via the localStorage convenience, and there's nothing left to do here —
      // don't silently reopen a finished job the recruiter didn't ask to revisit.
      clearStoredSession();
      sessionId = null;
      return;
    }
    setStoredSession(sessionId, data.job_state && data.job_state.job_title);
    currentPhase = data.phase;
    renderMessages(data.messages, data.asking_about_field);
    lastSavedAt = (data.job_record && data.job_record.updated_at) || null;
    updateSavedIndicator();
    renderJobPanel(data);
  } catch (err) {
    // Session no longer resolvable (404 or server error) — start fresh rather than block the recruiter.
    clearStoredSession();
    sessionId = null;
  }
}

init();
