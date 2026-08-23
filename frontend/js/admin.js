const tableBody = document.getElementById("admin-table-body");
const searchInput = document.getElementById("admin-search");
const statTotal = document.getElementById("stat-total");
const statPublished = document.getElementById("stat-published");
const statDrafts = document.getElementById("stat-drafts");
const listView = document.getElementById("list-view");
const detailView = document.getElementById("detail-view");
const detailBody = document.getElementById("detail-body");
const backLink = document.getElementById("back-link");
const companyView = document.getElementById("company-view");
const companyBody = document.getElementById("company-body");
const companyBackLink = document.getElementById("company-back-link");

const JOB_FIELD_LABELS = [
  ["job_title", "Job Title"],
  ["job_category", "Job Category"],
  ["experience", "Experience"],
  ["location", "Location"],
  ["work_mode", "Work Mode"],
  ["employment_type", "Employment Type"],
  ["education", "Education"],
  ["salary", "Salary"],
  ["additional_information", "Additional Info"],
];

const COMPANY_FIELD_LABELS = [
  ["industry", "Industry"],
  ["website", "Website"],
  ["headquarters", "Headquarters"],
  ["company_overview", "Company Overview"],
  ["company_culture", "Company Culture"],
  ["benefits", "Benefits"],
  ["work_life_balance", "Work-Life Balance"],
  ["why_join_us", "Why Join Us"],
];

const JD_TEXT_FIELDS = [
  ["job_summary", "Job Summary"],
  ["about_role", "About the Role"],
  ["company_overview", "Company Overview"],
  ["why_company", "Why Join"],
];

// Responsibilities/Required Skills/Preferred Skills come from the JOB record itself (the same
// job_state-backed columns the recruiter edits in the draft panel), not from selected_jd — the
// drafted document no longer carries its own separate copy of these three (see
// JobDescriptionDraft's comment on the backend) — so there is exactly one copy shown here.
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
const CUSTOM_QUESTIONS_FIELD = ["custom_questions", "Custom Questions"];

let allJobs = [];

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
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

function dateCell(iso, fallback) {
  const td = document.createElement("td");
  if (!iso) {
    td.textContent = fallback || "—";
    return td;
  }
  td.textContent = formatRelative(iso);
  td.title = formatAbsolute(iso);
  return td;
}

function renderTable(jobs) {
  clearChildren(tableBody);
  if (jobs.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 9;
    td.className = "empty-state";
    td.textContent = "No jobs match your search.";
    tr.appendChild(td);
    tableBody.appendChild(tr);
    return;
  }
  jobs.forEach((job) => {
    const tr = document.createElement("tr");
    tr.addEventListener("click", () => showDetail(job.id));

    const idTd = document.createElement("td");
    idTd.textContent = job.job_id || "—";

    const companyTd = document.createElement("td");
    const companyLink = document.createElement("a");
    companyLink.href = "#";
    companyLink.textContent = job.company_name;
    companyLink.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      showCompany(job.company_id);
    });
    companyTd.appendChild(companyLink);

    const titleTd = document.createElement("td");
    titleTd.textContent = job.job_title;
    const locationTd = document.createElement("td");
    locationTd.textContent = job.location || "—";
    const workModeTd = document.createElement("td");
    workModeTd.textContent = job.work_mode || "—";
    const statusTd = document.createElement("td");
    statusTd.appendChild(statusBadges(job));

    const actionsTd = document.createElement("td");
    const viewBtn = document.createElement("button");
    viewBtn.type = "button";
    viewBtn.className = "btn btn-small";
    viewBtn.textContent = "View";
    viewBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showDetail(job.id);
    });
    const companyBtn = document.createElement("button");
    companyBtn.type = "button";
    companyBtn.className = "btn btn-small";
    companyBtn.textContent = "Company";
    companyBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showCompany(job.company_id);
    });
    actionsTd.appendChild(viewBtn);
    actionsTd.appendChild(companyBtn);

    tr.appendChild(idTd);
    tr.appendChild(companyTd);
    tr.appendChild(titleTd);
    tr.appendChild(locationTd);
    tr.appendChild(workModeTd);
    tr.appendChild(statusTd);
    tr.appendChild(dateCell(job.published_at, "Not published"));
    tr.appendChild(dateCell(job.updated_at));
    tr.appendChild(actionsTd);
    tableBody.appendChild(tr);
  });
}

function applySearch() {
  const query = searchInput.value.trim().toLowerCase();
  if (!query) {
    renderTable(allJobs);
    return;
  }
  const filtered = allJobs.filter((job) =>
    [job.job_title, job.company_name, job.job_id]
      .filter(Boolean)
      .some((field) => field.toLowerCase().includes(query))
  );
  renderTable(filtered);
}

async function renderList() {
  clearChildren(tableBody);
  const loadingRow = document.createElement("tr");
  const loadingCell = document.createElement("td");
  loadingCell.colSpan = 9;
  loadingCell.className = "empty-state";
  loadingCell.textContent = "Loading jobs…";
  loadingRow.appendChild(loadingCell);
  tableBody.appendChild(loadingRow);

  try {
    allJobs = await api.getAdminJobs();
    const published = allJobs.filter((j) => j.status === "published").length;
    const drafts = allJobs.filter((j) => j.status === "draft").length;
    statTotal.textContent = allJobs.length;
    statPublished.textContent = published;
    statDrafts.textContent = drafts;
    applySearch();
  } catch (err) {
    clearChildren(tableBody);
    const errorRow = document.createElement("tr");
    const errorCell = document.createElement("td");
    errorCell.colSpan = 9;
    errorCell.className = "empty-state";
    errorCell.textContent = "Something went wrong loading jobs. Please refresh to try again.";
    errorRow.appendChild(errorCell);
    tableBody.appendChild(errorRow);
  }
}

function fieldRow(label, value) {
  const row = document.createElement("div");
  row.className = "profile-view-row";
  const labelEl = document.createElement("div");
  labelEl.className = "label";
  labelEl.textContent = label;
  const valueEl = document.createElement("div");
  valueEl.className = "value";
  valueEl.textContent = value || "Not specified";
  row.appendChild(labelEl);
  row.appendChild(valueEl);
  return row;
}

function cardHeading(text) {
  const h2 = document.createElement("h2");
  h2.textContent = text;
  h2.style.fontSize = "16px";
  h2.style.marginTop = "0";
  return h2;
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

function jobListRow(job) {
  const row = document.createElement("div");
  row.className = "job-list-row job-list-row-clickable";
  row.addEventListener("click", () => showDetail(job.id));

  const idCell = document.createElement("div");
  idCell.className = "job-id-cell";
  idCell.textContent = job.job_id || "—";

  const titleCell = document.createElement("div");
  titleCell.className = "job-title-cell";
  titleCell.textContent = job.job_title;

  const metaCell = document.createElement("div");
  metaCell.className = "job-meta-cell";
  metaCell.textContent = [job.location, job.work_mode].filter(Boolean).join(" · ") || "—";

  row.appendChild(idCell);
  row.appendChild(titleCell);
  row.appendChild(metaCell);
  row.appendChild(statusBadges(job));
  return row;
}

async function showDetail(id) {
  const url = new URL(window.location);
  url.searchParams.set("id", id);
  url.searchParams.delete("company_id");
  window.history.pushState({}, "", url);

  try {
    const job = await api.getAdminJob(id);
    clearChildren(detailBody);

    const header = document.createElement("div");
    header.className = "card";
    const title = document.createElement("h1");
    title.className = "page-title";
    title.style.marginBottom = "6px";
    title.textContent = job.job_title;
    header.appendChild(title);
    const meta = document.createElement("div");
    meta.className = "job-card-meta";
    meta.textContent = [job.job_id || "Draft (unpublished)", job.company_name, job.status].filter(Boolean).join(" — ");
    header.appendChild(meta);
    const companyBtn = document.createElement("button");
    companyBtn.type = "button";
    companyBtn.className = "btn";
    companyBtn.style.marginTop = "10px";
    companyBtn.textContent = `View ${job.company_name} →`;
    companyBtn.addEventListener("click", () => showCompany(job.company_id));
    header.appendChild(companyBtn);
    detailBody.appendChild(header);

    const jobCard = document.createElement("div");
    jobCard.className = "card";
    jobCard.appendChild(cardHeading("Structured Job Information"));
    JOB_FIELD_LABELS.forEach(([key, label]) => jobCard.appendChild(fieldRow(label, job[key])));
    jobCard.appendChild(fieldRow("Required Skills", (job.required_skills || []).join(", ")));
    jobCard.appendChild(fieldRow("Preferred Skills", (job.preferred_skills || []).join(", ")));
    jobCard.appendChild(fieldRow("Responsibilities", (job.responsibilities || []).join(", ")));
    detailBody.appendChild(jobCard);

    const datesCard = document.createElement("div");
    datesCard.className = "card";
    datesCard.appendChild(cardHeading("Status & Dates"));
    datesCard.appendChild(fieldRow("Status", job.status));
    if (job.status === "published") {
      datesCard.appendChild(fieldRow("Accepting Applications", job.accepting_applications ? "Yes" : "No — closed"));
    }
    datesCard.appendChild(fieldRow("Created", formatAbsolute(job.created_at)));
    datesCard.appendChild(fieldRow("Last Updated", formatAbsolute(job.updated_at)));
    datesCard.appendChild(fieldRow("Published", job.published_at ? formatAbsolute(job.published_at) : "Not published yet"));
    detailBody.appendChild(datesCard);

    const jdCard = document.createElement("div");
    jdCard.className = "card";
    jdCard.appendChild(cardHeading("Final Job Description"));
    if (job.selected_jd) {
      jdCard.appendChild(jdSections(job));
    } else {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "Not yet generated — this job is still a draft.";
      jdCard.appendChild(empty);
    }
    detailBody.appendChild(jdCard);

    // Other jobs from the same company
    const otherJobs = allJobs.filter((j) => j.company_id === job.company_id && j.id !== job.id);
    if (otherJobs.length > 0) {
      const otherCard = document.createElement("div");
      otherCard.className = "card";
      otherCard.appendChild(cardHeading(`Other Jobs from ${job.company_name}`));
      otherJobs.forEach((j) => otherCard.appendChild(jobListRow(j)));
      detailBody.appendChild(otherCard);
    }

    listView.style.display = "none";
    companyView.style.display = "none";
    detailView.style.display = "block";
  } catch (err) {
    detailBody.textContent = "Job not found.";
    listView.style.display = "none";
    detailView.style.display = "block";
  }
}

async function showCompany(companyId) {
  const url = new URL(window.location);
  url.searchParams.set("company_id", companyId);
  url.searchParams.delete("id");
  window.history.pushState({}, "", url);

  try {
    const data = await api.getAdminCompany(companyId);
    clearChildren(companyBody);

    const header = document.createElement("div");
    header.className = "card";
    const title = document.createElement("h1");
    title.className = "page-title";
    title.style.marginBottom = "6px";
    title.textContent = data.company.company_name;
    header.appendChild(title);
    const meta = document.createElement("div");
    meta.className = "job-card-meta";
    const publishedCount = data.jobs.filter((j) => j.status === "published").length;
    meta.textContent = `${data.jobs.length} job${data.jobs.length === 1 ? "" : "s"} posted · ${publishedCount} published`;
    header.appendChild(meta);
    companyBody.appendChild(header);

    const profileCard = document.createElement("div");
    profileCard.className = "card";
    profileCard.appendChild(cardHeading("Company Profile"));
    COMPANY_FIELD_LABELS.forEach(([key, label]) => profileCard.appendChild(fieldRow(label, data.company[key])));
    companyBody.appendChild(profileCard);

    const jobsCard = document.createElement("div");
    jobsCard.className = "card";
    jobsCard.appendChild(cardHeading("Jobs"));
    if (data.jobs.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No jobs posted yet.";
      jobsCard.appendChild(empty);
    } else {
      data.jobs.forEach((j) => jobsCard.appendChild(jobListRow(j)));
    }
    companyBody.appendChild(jobsCard);

    listView.style.display = "none";
    detailView.style.display = "none";
    companyView.style.display = "block";
  } catch (err) {
    companyBody.textContent = "Company not found.";
    listView.style.display = "none";
    companyView.style.display = "block";
  }
}

function showList() {
  const url = new URL(window.location);
  url.searchParams.delete("id");
  url.searchParams.delete("company_id");
  window.history.pushState({}, "", url);
  detailView.style.display = "none";
  companyView.style.display = "none";
  listView.style.display = "block";
}

backLink.addEventListener("click", (e) => {
  e.preventDefault();
  showList();
});

companyBackLink.addEventListener("click", (e) => {
  e.preventDefault();
  showList();
});

searchInput.addEventListener("input", applySearch);

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
  // Backend authorization is the real boundary (every /api/admin/* call independently
  // requires role=admin) — this is the UX-layer half: redirect a signed-out visitor to
  // /auth.html, and a signed-in-but-non-admin recruiter away entirely, before showing
  // platform-wide data in the page shell.
  try {
    const user = await api.getMe();
    if (user.role !== "admin") {
      window.location.href = "/auth.html";
      return;
    }
  } catch (err) {
    window.location.href = "/auth.html";
    return;
  }

  await renderList();
  const params = new URLSearchParams(window.location.search);
  const id = params.get("id");
  const companyId = params.get("company_id");
  if (id) {
    await showDetail(id);
  } else if (companyId) {
    await showCompany(companyId);
  }
}

init();
