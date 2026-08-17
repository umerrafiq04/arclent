const companyNameEl = document.getElementById("company-name");
const statTotal = document.getElementById("stat-total");
const statPublished = document.getElementById("stat-published");
const statDrafts = document.getElementById("stat-drafts");
const profileView = document.getElementById("profile-view");
const profileForm = document.getElementById("profile-form");
const editProfileBtn = document.getElementById("edit-profile-btn");
const cancelProfileBtn = document.getElementById("cancel-profile-btn");
const recentJobsEl = document.getElementById("recent-jobs");
const viewAllJobsBtn = document.getElementById("view-all-jobs-btn");
const recruiterAvatarEl = document.getElementById("recruiter-avatar");
const recruiterNameEl = document.getElementById("recruiter-name");
const recruiterMetaEl = document.getElementById("recruiter-meta");
const velocityActiveEl = document.getElementById("velocity-active");
const velocityActiveFillEl = document.getElementById("velocity-active-fill");
const velocityAcceptingEl = document.getElementById("velocity-accepting");
const velocityAcceptingFillEl = document.getElementById("velocity-accepting-fill");

let showAllJobs = false;
let latestJobs = [];

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

function jobAvatarLetter(job) {
  return (job.job_title || "?").trim().charAt(0).toUpperCase() || "?";
}

function renderRecentJobs(jobs) {
  clearChildren(recentJobsEl);
  if (jobs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No jobs yet — click \"+ Post a Job\" to create your first one.";
    recentJobsEl.appendChild(empty);
    return;
  }
  const visible = showAllJobs ? jobs : jobs.slice(0, 8);
  visible.forEach((job) => {
    const row = document.createElement("div");
    row.className = "neo-job-row";

    const avatar = document.createElement("span");
    avatar.className = "neo-job-avatar";
    avatar.textContent = jobAvatarLetter(job);
    row.appendChild(avatar);

    const main = document.createElement("div");
    main.className = "neo-job-main";

    const title = document.createElement("div");
    title.className = "neo-job-title";
    title.textContent = job.job_title;
    main.appendChild(title);

    const tags = document.createElement("div");
    tags.className = "neo-job-tags";

    if (job.employment_type) {
      const empPill = document.createElement("span");
      empPill.className = "neo-pill";
      empPill.textContent = job.employment_type;
      tags.appendChild(empPill);
    }

    const statusPill = document.createElement("span");
    if (job.status === "published") {
      statusPill.className = job.accepting_applications ? "neo-pill neo-pill-active" : "neo-pill neo-pill-closed";
      statusPill.textContent = job.accepting_applications ? "Active" : "Closed";
    } else {
      statusPill.className = "neo-pill";
      statusPill.textContent = "Draft";
    }
    tags.appendChild(statusPill);

    const time = document.createElement("span");
    time.className = "neo-job-time";
    time.title = formatAbsolute(job.updated_at);
    time.textContent = formatRelative(job.updated_at);
    tags.appendChild(time);

    main.appendChild(tags);
    row.appendChild(main);

    const actions = document.createElement("div");
    actions.className = "neo-job-actions";
    actions.appendChild(manageButton(job));

    const boostBtn = document.createElement("button");
    boostBtn.type = "button";
    boostBtn.className = "neo-btn neo-btn-small neo-btn-yellow";
    boostBtn.textContent = "Boost";
    boostBtn.title = "Coming soon";
    boostBtn.disabled = true;
    actions.appendChild(boostBtn);

    row.appendChild(actions);
    recentJobsEl.appendChild(row);
  });
}

function manageButton(job) {
  const wrap = document.createElement("div");
  wrap.style.position = "relative";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "neo-btn neo-btn-small";
  btn.textContent = "Manage ▾";

  const menu = document.createElement("div");
  menu.className = "neo-manage-menu";

  if (job.status === "published") {
    const viewLink = document.createElement("button");
    viewLink.type = "button";
    viewLink.textContent = "View listing";
    viewLink.addEventListener("click", () => {
      window.location.href = `/jobs.html?job_id=${encodeURIComponent(job.job_id)}`;
    });
    menu.appendChild(viewLink);
  }

  const editBtn = document.createElement("button");
  editBtn.type = "button";
  editBtn.textContent = job.status === "published" ? "Edit" : "Continue editing";
  editBtn.addEventListener("click", () => {
    menu.classList.remove("open");
    if (window.openJobModal) window.openJobModal(job.session_id);
  });
  menu.appendChild(editBtn);

  if (job.status === "published") {
    const toggleBtn = document.createElement("button");
    toggleBtn.type = "button";
    toggleBtn.textContent = job.accepting_applications ? "Stop Hiring" : "Reopen Hiring";
    toggleBtn.addEventListener("click", async () => {
      menu.classList.remove("open");
      const nowClosing = job.accepting_applications;
      const confirmed = window.confirm(
        nowClosing
          ? `Stop accepting applications for ${job.job_title} (${job.job_id})? The listing stays published — this only closes it to new applicants.`
          : `Reopen ${job.job_title} (${job.job_id}) to new applications?`
      );
      if (!confirmed) return;
      try {
        await api.setJobAcceptingApplications(job.session_id, !job.accepting_applications);
        showToast(nowClosing ? "Applications closed for this job." : "Job reopened to applications.", "success");
        await loadJobs();
      } catch (err) {
        showToast(err.message || "Couldn't update hiring status. Please try again.", "error");
      }
    });
    menu.appendChild(toggleBtn);
  }

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "danger";
  deleteBtn.textContent = "Delete";
  deleteBtn.addEventListener("click", async () => {
    menu.classList.remove("open");
    const confirmed = window.confirm(`Permanently delete "${job.job_title}"? This can't be undone.`);
    if (!confirmed) return;
    try {
      await api.deleteJob(job.session_id);
      showToast("Job deleted.", "success");
      await loadJobs();
    } catch (err) {
      showToast(err.message || "Couldn't delete this job. Please try again.", "error");
    }
  });
  menu.appendChild(deleteBtn);

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const wasOpen = menu.classList.contains("open");
    document.querySelectorAll(".neo-manage-menu.open").forEach((m) => m.classList.remove("open"));
    if (!wasOpen) menu.classList.add("open");
  });

  wrap.appendChild(btn);
  wrap.appendChild(menu);
  return wrap;
}

document.addEventListener("click", () => {
  document.querySelectorAll(".neo-manage-menu.open").forEach((m) => m.classList.remove("open"));
});

if (viewAllJobsBtn) {
  viewAllJobsBtn.addEventListener("click", () => {
    showAllJobs = !showAllJobs;
    viewAllJobsBtn.textContent = showAllJobs ? "Show Less" : "View All";
    renderRecentJobs(latestJobs);
  });
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
  latestJobs = jobs;
  const published = jobs.filter((j) => j.status === "published");
  const drafts = jobs.filter((j) => j.status === "draft").length;
  const accepting = published.filter((j) => j.accepting_applications).length;
  statTotal.textContent = jobs.length;
  statPublished.textContent = published.length;
  statDrafts.textContent = drafts;

  // Real, computable numbers only — no fabricated proposal/hire counts (that data doesn't
  // exist in this schema yet).
  velocityActiveEl.textContent = `${published.length} / ${jobs.length}`;
  velocityActiveFillEl.style.width = jobs.length ? `${Math.round((published.length / jobs.length) * 100)}%` : "0%";
  velocityAcceptingEl.textContent = published.length ? `${accepting} / ${published.length}` : "—";
  velocityAcceptingFillEl.style.width = published.length ? `${Math.round((accepting / published.length) * 100)}%` : "0%";

  renderRecentJobs(jobs);
}

editProfileBtn.addEventListener("click", () => {
  profileView.style.display = "none";
  editProfileBtn.style.display = "none";
  // Clear the inline override rather than hardcoding "flex" — an inline style always beats
  // the stylesheet regardless of specificity, which was silently defeating the responsive
  // .profile-form-grid 2-column layout. Clearing it lets styles.css decide (grid ≥720px).
  profileForm.style.display = "";
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

const postJobBtn = document.getElementById("post-job-btn");
if (postJobBtn) {
  postJobBtn.addEventListener("click", () => {
    if (window.openJobModal) window.openJobModal();
  });
}

async function init() {
  // Backend authorization is the real boundary (every /api/* call below re-checks it) — this
  // is purely so an unauthenticated visitor isn't shown an empty dashboard shell first.
  let me;
  try {
    me = await api.getMe();
  } catch (err) {
    window.location.href = "/auth.html";
    return;
  }

  if (me && me.name) {
    recruiterNameEl.textContent = me.name;
    recruiterAvatarEl.textContent = me.name.trim().charAt(0).toUpperCase() || "?";
    recruiterMetaEl.textContent = me.company_name || "Arclent Member";
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
