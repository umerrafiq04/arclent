const jobsGrid = document.getElementById("jobs-grid");
const listView = document.getElementById("list-view");
const detailView = document.getElementById("detail-view");
const detailBody = document.getElementById("detail-body");
const backLink = document.getElementById("back-link");
const applyModalOverlay = document.getElementById("apply-modal-overlay");
const applyModalClose = document.getElementById("apply-modal-close");

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

// Recruiter-typed screening questions — job_state-backed like JOB_LIST_FIELDS above, but rendered
// last (after Benefits) rather than grouped with those, matching the draft panel's own section
// order. Never AI-touched (see JobState.custom_questions on the backend).
const CUSTOM_QUESTIONS_FIELD = ["custom_questions", "Questions"];

// Stacked label/value fields shown right under the title, before the full description body —
// only the ones the job actually has get rendered.
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

function detailMetaGrid(job) {
  const grid = document.createElement("div");
  grid.className = "jd-meta-grid";
  DETAIL_META_FIELDS.forEach(([key, label]) => {
    const value = job[key];
    if (!value) return;
    const item = document.createElement("div");
    item.className = "jd-meta-item";
    const labelEl = document.createElement("div");
    labelEl.className = "jd-meta-label";
    labelEl.textContent = label;
    const valueEl = document.createElement("div");
    valueEl.className = "jd-meta-value";
    valueEl.textContent = value;
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
  const [cqKey, cqLabel] = CUSTOM_QUESTIONS_FIELD;
  const customQuestions = job[cqKey];
  if (customQuestions && customQuestions.length > 0) {
    frag.appendChild(jdListField(cqLabel, customQuestions));
  }
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
      applyBtn.addEventListener("click", () => {
        applyModalOverlay.style.display = "flex";
      });
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

applyModalClose.addEventListener("click", () => {
  applyModalOverlay.style.display = "none";
});

applyModalOverlay.addEventListener("click", (e) => {
  if (e.target === applyModalOverlay) applyModalOverlay.style.display = "none";
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
