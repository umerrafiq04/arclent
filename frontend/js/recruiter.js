const companyNameEl = document.getElementById("company-name");
const statTotal = document.getElementById("stat-total");
const statPublished = document.getElementById("stat-published");
const statDrafts = document.getElementById("stat-drafts");
const profileView = document.getElementById("profile-view");
const profileForm = document.getElementById("profile-form");
const editProfileBtn = document.getElementById("edit-profile-btn");
const cancelProfileBtn = document.getElementById("cancel-profile-btn");
const recentJobsEl = document.getElementById("recent-jobs");

const PROFILE_FIELDS = [
  ["industry", "Industry"],
  ["website", "Website"],
  ["headquarters", "Headquarters"],
  ["company_overview", "Overview"],
  ["company_culture", "Culture"],
  ["benefits", "Benefits"],
  ["work_life_balance", "Work-Life Balance"],
  ["why_join_us", "Why Join Us"],
];

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function renderProfileView(profile) {
  clearChildren(profileView);
  PROFILE_FIELDS.forEach(([key, label]) => {
    const row = document.createElement("div");
    row.className = "profile-view-row";
    const labelEl = document.createElement("div");
    labelEl.className = "label";
    labelEl.textContent = label;
    const valueEl = document.createElement("div");
    valueEl.className = "value";
    valueEl.textContent = profile[key] || "Not set";
    row.appendChild(labelEl);
    row.appendChild(valueEl);
    profileView.appendChild(row);
  });
}

function populateForm(profile) {
  Object.entries(profile).forEach(([key, value]) => {
    const field = profileForm.elements.namedItem(key);
    if (field) field.value = value || "";
  });
}

function statusBadges(job) {
  const frag = document.createDocumentFragment();
  const badge = document.createElement("span");
  if (job.status === "published") {
    badge.className = "badge badge-published";
    badge.textContent = "Published";
  } else {
    badge.className = "badge badge-draft";
    badge.textContent = "Draft";
  }
  frag.appendChild(badge);

  if (job.status === "published" && !job.accepting_applications) {
    const closedBadge = document.createElement("span");
    closedBadge.className = "badge badge-closed";
    closedBadge.textContent = "Applications Closed";
    frag.appendChild(closedBadge);
  }
  return frag;
}

function renderRecentJobs(jobs) {
  clearChildren(recentJobsEl);
  if (jobs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No jobs yet — click \"Post a New Job\" to create your first one.";
    recentJobsEl.appendChild(empty);
    return;
  }
  jobs.slice(0, 8).forEach((job) => {
    const row = document.createElement("div");
    row.className = "job-list-row";

    const idCell = document.createElement("div");
    idCell.className = "job-id-cell";
    idCell.textContent = job.job_id || "—";

    const titleCell = document.createElement("div");
    titleCell.className = "job-title-cell";
    titleCell.textContent = job.job_title;

    const metaCell = document.createElement("div");
    metaCell.className = "job-meta-cell";
    metaCell.textContent = [job.location, job.work_mode].filter(Boolean).join(" · ") || "—";

    const dateCell = document.createElement("div");
    dateCell.className = "job-meta-cell";
    dateCell.title = formatAbsolute(job.updated_at);
    dateCell.textContent = formatRelative(job.updated_at);

    const actionsCell = document.createElement("div");
    actionsCell.className = "job-actions-cell";
    if (job.status === "published") {
      const viewLink = document.createElement("a");
      viewLink.className = "btn btn-small";
      viewLink.textContent = "View";
      viewLink.href = `/jobs.html?job_id=${encodeURIComponent(job.job_id)}`;
      const editLink = document.createElement("a");
      editLink.className = "btn btn-small";
      editLink.textContent = "Edit Job";
      editLink.href = `/create-job.html?session_id=${encodeURIComponent(job.session_id)}`;
      actionsCell.appendChild(viewLink);
      actionsCell.appendChild(editLink);
      actionsCell.appendChild(hiringToggleButton(job));
    } else {
      const continueLink = document.createElement("a");
      continueLink.className = "btn btn-small btn-primary";
      continueLink.textContent = "Continue";
      continueLink.href = `/create-job.html?session_id=${encodeURIComponent(job.session_id)}`;
      actionsCell.appendChild(continueLink);
    }

    row.appendChild(idCell);
    row.appendChild(titleCell);
    row.appendChild(metaCell);
    row.appendChild(dateCell);
    row.appendChild(statusBadges(job));
    row.appendChild(actionsCell);
    recentJobsEl.appendChild(row);
  });
}

function hiringToggleButton(job) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn btn-small";
  btn.textContent = job.accepting_applications ? "Stop Hiring" : "Reopen Hiring";
  btn.addEventListener("click", async () => {
    const nowClosing = job.accepting_applications;
    const confirmed = window.confirm(
      nowClosing
        ? `Stop accepting applications for ${job.job_title} (${job.job_id})? The listing stays published — this only closes it to new applicants.`
        : `Reopen ${job.job_title} (${job.job_id}) to new applications?`
    );
    if (!confirmed) return;
    btn.disabled = true;
    try {
      await api.setJobAcceptingApplications(job.session_id, !job.accepting_applications);
      showToast(nowClosing ? "Applications closed for this job." : "Job reopened to applications.", "success");
      await loadJobs();
    } catch (err) {
      showToast(err.message || "Couldn't update hiring status. Please try again.", "error");
      btn.disabled = false;
    }
  });
  return btn;
}

let currentProfile = null;

async function loadProfile() {
  currentProfile = await api.getCompanyProfile();
  companyNameEl.textContent = currentProfile.company_name;
  renderProfileView(currentProfile);
  populateForm(currentProfile);
}

async function loadJobs() {
  const jobs = await api.getJobs();
  const published = jobs.filter((j) => j.status === "published").length;
  const drafts = jobs.filter((j) => j.status === "draft").length;
  statTotal.textContent = jobs.length;
  statPublished.textContent = published;
  statDrafts.textContent = drafts;
  renderRecentJobs(jobs);
}

editProfileBtn.addEventListener("click", () => {
  profileView.style.display = "none";
  editProfileBtn.style.display = "none";
  profileForm.style.display = "flex";
});

cancelProfileBtn.addEventListener("click", () => {
  populateForm(currentProfile);
  profileForm.style.display = "none";
  profileView.style.display = "block";
  editProfileBtn.style.display = "inline-flex";
});

profileForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const saveBtn = profileForm.querySelector('button[type="submit"]');
  const formData = new FormData(profileForm);
  const updates = Object.fromEntries(formData.entries());

  saveBtn.disabled = true;
  saveBtn.textContent = "Saving…";
  try {
    currentProfile = await api.putCompanyProfile(updates);
    companyNameEl.textContent = currentProfile.company_name;
    renderProfileView(currentProfile);
    profileForm.style.display = "none";
    profileView.style.display = "block";
    editProfileBtn.style.display = "inline-flex";
    showToast("Company profile saved.", "success");
  } catch (err) {
    showToast(err.message || "Failed to save company profile. Please try again.", "error");
  } finally {
    saveBtn.disabled = false;
    saveBtn.textContent = "Save Company Profile";
  }
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
  // Backend authorization is the real boundary (every /api/* call below re-checks it) — this
  // is purely so an unauthenticated visitor isn't shown an empty dashboard shell first.
  try {
    await api.getMe();
  } catch (err) {
    window.location.href = "/auth.html";
    return;
  }

  const loading = document.createElement("div");
  loading.className = "empty-state";
  loading.textContent = "Loading jobs…";
  recentJobsEl.appendChild(loading);

  try {
    await Promise.all([loadProfile(), loadJobs()]);
  } catch (err) {
    companyNameEl.textContent = "Something went wrong loading the dashboard";
    showToast("Couldn't load the dashboard. Please refresh to try again.", "error");
    console.error(err);
  }
}

init();
