const jobsGrid = document.getElementById("jobs-grid");
const listView = document.getElementById("list-view");
const detailView = document.getElementById("detail-view");
const detailBody = document.getElementById("detail-body");
const backLink = document.getElementById("back-link");
const applyModalOverlay = document.getElementById("apply-modal-overlay");
const applyModalClose = document.getElementById("apply-modal-close");
const applyForm = document.getElementById("apply-form");
const applyQuestionsContainer = document.getElementById("apply-questions");
const applyFormError = document.getElementById("apply-form-error");
const applySubmitBtn = document.getElementById("apply-submit-btn");
const applySuccess = document.getElementById("apply-success");
const applySuccessClose = document.getElementById("apply-success-close");

let applyJobId = null;

const JD_TEXT_FIELDS = [
  ["job_summary", "Job Summary"],
  ["about_role", "About the Role"],
  ["company_overview", "Company Overview"],
  ["why_company", "Why Join"],
];

// Responsibilities/Required Skills/Preferred Skills come from the JOB record itself (the same
// job_state-backed columns the recruiter edits in the draft panel), not from selected_jd — the
// drafted document no longer carries its own separate copy of these three (see
// JobDescriptionDraft's comment on the backend) — so there is exactly one copy shown here, always
// matching what the recruiter's panel shows and what Regenerate treats as the source of truth.
const JOB_LIST_FIELDS = [
  ["responsibilities", "Responsibilities"],
  ["required_skills", "Required Skills"],
  ["preferred_skills", "Preferred Skills"],
];

const JD_LIST_FIELDS = [
  ["stand_out", "Ways to Stand Out"],
  ["benefits", "Benefits / Employee Experience"],
];

// Stacked label/value fields shown right under the title, before the full description body —
// only the ones the job actually has get rendered. Platforms is shown separately, as logo
// badges (see PLATFORM_ICONS/platformIconsRow below), never as a text row here and never inside
// jdSections below — it's recruitment targeting, not job description content.
const DETAIL_META_FIELDS = [
  ["company_name", "Company"],
  ["job_id", "Job Requisition ID"],
  ["job_category", "Job Category"],
  ["employment_type", "Time Type"],
  ["location", "Locations"],
  ["work_mode", "Work Mode"],
  ["experience", "Experience"],
  ["education", "Education"],
  ["salary", "Salary"],
];

// Small brand-colored logo badges for the platforms checklist field — shown instead of a text
// list, on both the Available Jobs cards and the View Job detail page. Approximate marks (this
// app has no official brand asset license), built as inline SVG so nothing depends on an external
// icon CDN. Keyed by the exact strings the chat checklist / draft panel dropdown use.
const PLATFORM_ICONS = {
  Facebook:
    '<svg viewBox="0 0 24 24" width="18" height="18"><circle cx="12" cy="12" r="12" fill="#1877F2"/><path d="M13.5 21v-7.2h2.4l.36-2.8h-2.76v-1.8c0-.81.22-1.36 1.39-1.36h1.48V5.34C15.9 5.24 15.02 5.16 14 5.16c-2.28 0-3.84 1.39-3.84 3.94v2.06H7.8v2.8h2.36V21h3.34z" fill="#fff"/></svg>',
  YouTube:
    '<svg viewBox="0 0 24 24" width="18" height="18"><rect width="24" height="24" rx="6" fill="#FF0000"/><path d="M10 8.3l6.2 3.7-6.2 3.7V8.3z" fill="#fff"/></svg>',
  Instagram:
    '<svg viewBox="0 0 24 24" width="18" height="18"><defs><linearGradient id="igGrad" x1="0" y1="1" x2="1" y2="0"><stop offset="0%" stop-color="#feda75"/><stop offset="35%" stop-color="#d62976"/><stop offset="70%" stop-color="#962fbf"/><stop offset="100%" stop-color="#4f5bd5"/></linearGradient></defs><rect width="24" height="24" rx="6" fill="url(#igGrad)"/><rect x="6" y="6" width="12" height="12" rx="3.5" fill="none" stroke="#fff" stroke-width="1.6"/><circle cx="12" cy="12" r="3.1" fill="none" stroke="#fff" stroke-width="1.6"/><circle cx="16.3" cy="7.7" r="1" fill="#fff"/></svg>',
  TikTok:
    '<svg viewBox="0 0 24 24" width="18" height="18"><rect width="24" height="24" rx="6" fill="#000"/><path d="M15.8 6.3c.3 1.6 1.3 2.6 3 2.8v2.1c-1.1 0-2.1-.3-3-1v4.5a4.1 4.1 0 1 1-4.1-4.1c.2 0 .4 0 .6.1v2.2a2 2 0 1 0 1.5 1.9V6.3h2z" fill="#25F4EE" transform="translate(0.5,-0.4)"/><path d="M15.8 6.3c.3 1.6 1.3 2.6 3 2.8v2.1c-1.1 0-2.1-.3-3-1v4.5a4.1 4.1 0 1 1-4.1-4.1c.2 0 .4 0 .6.1v2.2a2 2 0 1 0 1.5 1.9V6.3h2z" fill="#FE2C55" transform="translate(-0.5,0.4)"/><path d="M15.8 6.3c.3 1.6 1.3 2.6 3 2.8v2.1c-1.1 0-2.1-.3-3-1v4.5a4.1 4.1 0 1 1-4.1-4.1c.2 0 .4 0 .6.1v2.2a2 2 0 1 0 1.5 1.9V6.3h2z" fill="#fff"/></svg>',
  Vimeo:
    '<svg viewBox="0 0 24 24" width="18" height="18"><rect width="24" height="24" rx="6" fill="#1AB7EA"/><path d="M19 8.6c-.1 1.9-1.4 4.6-4 8-2.7 3.6-5 5.4-6.8 5.4-1.1 0-2.1-1-2.9-3.1L4 13.4c-.5-1.7-1-1.7-1.9-1.2l-.6.4-.5-.7 2.7-2.4C4.8 8.6 5.7 7.9 6.4 7.9c1.5-.1 2.4.9 2.8 3.1.5 2.4.8 3.9 1 4.5.5 1.5 1.1 2.2 1.7 2.2.5 0 1.2-.6 2.1-1.9.9-1.3 1.4-2.2 1.5-2.9.1-1.1-.4-1.6-1.5-1.6-.5 0-1.1.1-1.6.3 1.1-3.5 3.2-5.2 6.2-5.1 2.3.1 3.3 1.5 3.2 4.1z" fill="#fff"/></svg>',
  Twitch:
    '<svg viewBox="0 0 24 24" width="18" height="18"><rect width="24" height="24" rx="6" fill="#9146FF"/><path d="M7 5.5L5.5 9v9h3.2V20l2-1.5h2.4L17 15V5.5H7zm8.5 8.6l-1.9 1.9h-2.4l-1.7 1.7v-1.7H7.1V6.6h8.4v7.5z" fill="#fff"/><rect x="10.3" y="8.5" width="1.2" height="3.2" fill="#9146FF"/><rect x="13.3" y="8.5" width="1.2" height="3.2" fill="#9146FF"/></svg>',
  Discord:
    '<svg viewBox="0 0 24 24" width="18" height="18"><rect width="24" height="24" rx="6" fill="#5865F2"/><ellipse cx="9" cy="13" rx="1.5" ry="1.8" fill="#fff"/><ellipse cx="15" cy="13" rx="1.5" ry="1.8" fill="#fff"/><path d="M7 8.5c1.5-.8 3.2-1.2 5-1.2s3.5.4 5 1.2" stroke="#fff" stroke-width="1.2" fill="none" stroke-linecap="round"/></svg>',
};

// Renders the small logo-badge row for a job's platforms — returns null (append-safe as a no-op)
// when the job has none, so every call site can unconditionally appendChild without an extra
// guard. Each badge carries a title tooltip with the platform name for accessibility/clarity.
// The Instagram mark's gradient has a fixed "igGrad" id in PLATFORM_ICONS — SVG ids must be
// unique document-wide, so with several Instagram badges on one page (multiple job cards, or a
// list card plus the detail page) later ones silently fail to resolve the gradient and render
// invisible. Give every badge instance its own id via a simple counter before injecting it.
let _platformIconInstanceCounter = 0;
function platformIconsRow(platforms) {
  if (!platforms || platforms.length === 0) return null;
  const row = document.createElement("div");
  row.className = "platform-icons-row";
  platforms.forEach((name) => {
    const svgMarkup = PLATFORM_ICONS[name];
    if (!svgMarkup) return;
    const uniqueId = `igGrad${_platformIconInstanceCounter++}`;
    const badge = document.createElement("span");
    badge.className = "platform-icon-badge";
    badge.title = name;
    badge.innerHTML = svgMarkup.replaceAll("igGrad", uniqueId);
    row.appendChild(badge);
  });
  return row;
}

function detailMetaGrid(job) {
  const grid = document.createElement("div");
  grid.className = "jd-meta-grid";
  DETAIL_META_FIELDS.forEach(([key, label]) => {
    const value = job[key];
    if (!value || (Array.isArray(value) && value.length === 0)) return;
    const item = document.createElement("div");
    item.className = "jd-meta-item";
    const labelEl = document.createElement("div");
    labelEl.className = "jd-meta-label";
    labelEl.textContent = label;
    const valueEl = document.createElement("div");
    valueEl.className = "jd-meta-value";
    valueEl.textContent = Array.isArray(value) ? value.join(", ") : value;
    item.appendChild(labelEl);
    item.appendChild(valueEl);
    grid.appendChild(item);
  });
  return grid;
}

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function jobCard(job) {
  const card = document.createElement("div");
  card.className = "job-card";

  const left = document.createElement("div");
  const title = document.createElement("div");
  title.className = "job-card-title";
  title.textContent = job.job_title;
  const meta = document.createElement("div");
  meta.className = "job-card-meta";
  meta.textContent = [job.company_name, [job.location, job.work_mode].filter(Boolean).join(" · "), job.experience]
    .filter(Boolean)
    .join(" — ");
  const idLine = document.createElement("div");
  idLine.className = "job-card-meta";
  idLine.textContent = job.job_id;
  left.appendChild(title);
  left.appendChild(meta);
  const platformIcons = platformIconsRow(job.platforms);
  if (platformIcons) left.appendChild(platformIcons);
  left.appendChild(idLine);

  if (!job.accepting_applications) {
    const closedBadge = document.createElement("span");
    closedBadge.className = "badge badge-closed";
    closedBadge.textContent = "Applications Closed";
    closedBadge.style.marginTop = "6px";
    left.appendChild(closedBadge);
  }

  if (job.published_at) {
    const postedLine = document.createElement("div");
    postedLine.className = "job-card-posted";
    postedLine.textContent = `Posted ${formatRelative(job.published_at)}`;
    postedLine.title = formatAbsolute(job.published_at);
    left.appendChild(postedLine);
  }

  const viewBtn = document.createElement("a");
  viewBtn.className = "btn btn-primary";
  viewBtn.textContent = "View Job";
  viewBtn.href = `?job_id=${encodeURIComponent(job.job_id)}`;
  viewBtn.addEventListener("click", (e) => {
    e.preventDefault();
    showDetail(job.job_id);
  });

  card.appendChild(left);
  card.appendChild(viewBtn);
  return card;
}

async function renderList() {
  clearChildren(jobsGrid);
  const loading = document.createElement("div");
  loading.className = "empty-state";
  loading.textContent = "Loading jobs…";
  jobsGrid.appendChild(loading);

  try {
    const jobs = await api.getPublicJobs();
    clearChildren(jobsGrid);
    if (jobs.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No published jobs yet.";
      jobsGrid.appendChild(empty);
      return;
    }
    jobs.forEach((job) => jobsGrid.appendChild(jobCard(job)));
  } catch (err) {
    clearChildren(jobsGrid);
    const errorEl = document.createElement("div");
    errorEl.className = "empty-state";
    errorEl.textContent = "Something went wrong loading jobs. Please refresh to try again.";
    jobsGrid.appendChild(errorEl);
  }
}

function jdListField(label, items) {
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
  return field;
}

// Takes the FULL job record (not just selected_jd) — Responsibilities/Required Skills/Preferred
// Skills render from the job's own job_state-backed columns (JOB_LIST_FIELDS), the rest from the
// drafted document (job.selected_jd).
function jdSections(job) {
  const jd = job.selected_jd || {};
  const frag = document.createDocumentFragment();
  JD_TEXT_FIELDS.forEach(([key, label]) => {
    if (!jd[key]) return;
    const field = document.createElement("div");
    field.className = "jd-field";
    const h4 = document.createElement("h4");
    h4.textContent = label;
    const p = document.createElement("p");
    p.textContent = jd[key];
    field.appendChild(h4);
    field.appendChild(p);
    frag.appendChild(field);
  });
  JOB_LIST_FIELDS.forEach(([key, label]) => {
    const items = job[key];
    if (!items || items.length === 0) return;
    frag.appendChild(jdListField(label, items));
  });
  JD_LIST_FIELDS.forEach(([key, label]) => {
    const items = jd[key];
    if (!items || items.length === 0) return;
    frag.appendChild(jdListField(label, items));
  });
  // Screening questions are deliberately NOT shown here — they belong to the Apply flow, where a
  // candidate answers them directly, not to the public job description itself. See applyModal.
  return frag;
}

async function showDetail(jobId) {
  const url = new URL(window.location);
  url.searchParams.set("job_id", jobId);
  window.history.pushState({}, "", url);

  try {
    const job = await api.getPublicJob(jobId);
    clearChildren(detailBody);

    const card = document.createElement("div");
    card.className = "card";

    const title = document.createElement("h1");
    title.className = "page-title";
    title.style.marginBottom = "6px";
    title.textContent = job.job_title;
    card.appendChild(title);

    const platformIcons = platformIconsRow(job.platforms);
    if (platformIcons) {
      platformIcons.classList.add("platform-icons-row-lg");
      card.appendChild(platformIcons);
    }

    if (job.published_at) {
      const postedLine = document.createElement("div");
      postedLine.className = "job-card-posted";
      postedLine.textContent = `Posted ${formatRelative(job.published_at)}`;
      postedLine.title = formatAbsolute(job.published_at);
      card.appendChild(postedLine);
    }

    if (!job.accepting_applications) {
      const closedBadge = document.createElement("span");
      closedBadge.className = "badge badge-closed";
      closedBadge.textContent = "Applications Closed";
      closedBadge.style.marginTop = "8px";
      card.appendChild(closedBadge);
    }

    const sectionTitle = document.createElement("div");
    sectionTitle.className = "jd-section-title";
    sectionTitle.style.marginTop = "18px";
    sectionTitle.textContent = "Job Description";
    card.appendChild(sectionTitle);

    card.appendChild(detailMetaGrid(job));

    card.appendChild(jdSections(job));

    const applyRow = document.createElement("div");
    applyRow.className = "apply-row";
    const applyBtn = document.createElement("button");
    applyBtn.type = "button";
    if (job.accepting_applications) {
      applyBtn.className = "btn btn-primary";
      applyBtn.textContent = "Apply Now";
      applyBtn.addEventListener("click", () => openApplyModal(job));
    } else {
      applyBtn.className = "btn";
      applyBtn.textContent = "Applications Closed";
      applyBtn.disabled = true;
    }
    applyRow.appendChild(applyBtn);
    card.appendChild(applyRow);

    detailBody.appendChild(card);
    listView.style.display = "none";
    detailView.style.display = "block";
  } catch (err) {
    detailBody.textContent = "Job not found.";
    listView.style.display = "none";
    detailView.style.display = "block";
  }
}

function showList() {
  const url = new URL(window.location);
  url.searchParams.delete("job_id");
  window.history.pushState({}, "", url);
  detailView.style.display = "none";
  listView.style.display = "block";
}

backLink.addEventListener("click", (e) => {
  e.preventDefault();
  showList();
});

// Screening questions the recruiter added (job.custom_questions — see JobState.custom_questions
// on the backend) render here as answerable inputs, one per question, keyed by the exact question
// text (no stable id of its own, see JobApplicationRequest on the backend) — this is the ONE place
// they're shown to a candidate; the public job description itself deliberately never displays them
// (see jdSections above).
function openApplyModal(job) {
  applyJobId = job.job_id;
  applyForm.reset();
  clearChildren(applyQuestionsContainer);
  (job.custom_questions || []).forEach((question, index) => {
    const label = document.createElement("label");
    label.textContent = question;
    const input = document.createElement("input");
    input.type = "text";
    input.dataset.question = question;
    input.id = `apply-question-${index}`;
    label.appendChild(input);
    applyQuestionsContainer.appendChild(label);
  });
  applyFormError.style.display = "none";
  applyForm.style.display = "flex";
  applySuccess.style.display = "none";
  applyModalOverlay.style.display = "flex";
}

function closeApplyModal() {
  applyModalOverlay.style.display = "none";
}

applyModalClose.addEventListener("click", closeApplyModal);
applySuccessClose.addEventListener("click", closeApplyModal);

applyModalOverlay.addEventListener("click", (e) => {
  if (e.target === applyModalOverlay) closeApplyModal();
});

applyForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  applyFormError.style.display = "none";
  const answers = {};
  applyQuestionsContainer.querySelectorAll("input[data-question]").forEach((input) => {
    if (input.value.trim()) answers[input.dataset.question] = input.value.trim();
  });
  applySubmitBtn.disabled = true;
  const originalText = applySubmitBtn.textContent;
  applySubmitBtn.textContent = "Submitting…";
  try {
    await api.applyToJob(applyJobId, { answers });
    applyForm.style.display = "none";
    applySuccess.style.display = "block";
  } catch (err) {
    applyFormError.textContent = err.message || "Couldn't submit your application. Please try again.";
    applyFormError.style.display = "block";
  } finally {
    applySubmitBtn.disabled = false;
    applySubmitBtn.textContent = originalText;
  }
});

async function adjustNavForCurrentUser() {
  // Public page, no auth gate — but an admin has no "Dashboard" of their own, and without
  // this there was no way back to /admin.html once they came here from it. Anonymous
  // visitors keep the default Dashboard link and no Log out entry (there's no session to
  // log out of); a signed-in recruiter or admin gets the same three-item nav — Dashboard/
  // Admin, Public Jobs, Log out — as every other page.
  const dashboardLink = document.getElementById("dashboard-nav-link");
  const logoutBtn = document.getElementById("logout-btn");
  try {
    const user = await api.getMe();
    if (user.role === "admin") {
      dashboardLink.textContent = "Admin";
      dashboardLink.href = "/admin.html";
    }
    logoutBtn.style.display = "";
    logoutBtn.addEventListener("click", async () => {
      try {
        await api.logout();
      } catch (_) {
        // Ignore — redirecting to /auth.html either way makes the client-side state moot.
      }
      window.location.href = "/auth.html";
    });
  } catch (err) {
    // Not signed in — leave the default Dashboard link as-is, Log out stays hidden.
  }
}

async function init() {
  adjustNavForCurrentUser();
  await renderList();
  const params = new URLSearchParams(window.location.search);
  const jobId = params.get("job_id");
  if (jobId) {
    await showDetail(jobId);
  }
}

init();
