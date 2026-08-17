(function () {
  const overlay = document.getElementById("job-modal-overlay");
  if (!overlay) return; // not on this page

  const closeBtn = document.getElementById("job-modal-close");
  const statusEl = document.getElementById("job-modal-status");
  const progressEl = document.getElementById("job-modal-progress");

  const messageList = document.getElementById("jm-message-list");
  const errorSlot = document.getElementById("jm-error-slot");
  const attachmentSlot = document.getElementById("jm-attachment-slot");
  const chatForm = document.getElementById("jm-chat-form");
  const chatInput = document.getElementById("jm-chat-input");
  const sendBtn = document.getElementById("jm-send-btn");
  const attachBtn = document.getElementById("jm-attach-btn");
  const fileInput = document.getElementById("jm-file-input");

  const draftEmpty = document.getElementById("jm-draft-empty");
  const draftForm = document.getElementById("jm-draft-form");

  const STORAGE_KEY = "recruiter_active_job_session";

  let sessionId = null;
  let currentPhase = null;
  let pendingFile = null;
  let currentData = null;
  let viewVersion = "1";

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function avatarEl() {
    const a = document.createElement("div");
    a.className = "avatar";
    a.textContent = "A";
    return a;
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

  function appendChips(afterRow, options) {
    if (!options || options.length === 0) return;
    const row = document.createElement("div");
    row.className = "jm-chip-row";
    options.forEach((opt) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "jm-chip";
      chip.textContent = opt;
      chip.addEventListener("click", () => {
        row.querySelectorAll(".jm-chip").forEach((c) => (c.disabled = true));
        sendMessage(opt);
      });
      row.appendChild(chip);
    });
    afterRow.insertAdjacentElement("afterend", row);
    messageList.scrollTop = messageList.scrollHeight;
  }

  function showProcessingStatus(label) {
    let row = document.getElementById("jm-processing-status");
    if (!row) {
      row = document.createElement("div");
      row.className = "message-row ai";
      row.id = "jm-processing-status";
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
    const el = document.getElementById("jm-processing-status");
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

  function renderMessages(messages, suggestedOptions) {
    clearChildren(messageList);
    if (!messages || messages.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "What are you hiring for today? 👋";
      messageList.appendChild(empty);
      return;
    }
    let lastAiRow = null;
    messages.forEach((m) => {
      if (m.jd_document) return; // JD content lives in the draft pane, not the chat, here
      const row = appendMessage(m.role, m.content);
      if (m.role !== "user") lastAiRow = row;
    });
    if (lastAiRow && suggestedOptions && suggestedOptions.length) {
      appendChips(lastAiRow, suggestedOptions);
    }
  }

  function fieldRow(labelText, value, onCommit, opts) {
    opts = opts || {};
    const wrap = document.createElement("label");
    wrap.className = "jm-field";
    const label = document.createElement("span");
    label.className = "jm-field-label";
    label.textContent = labelText;
    wrap.appendChild(label);

    let input;
    if (opts.select) {
      input = document.createElement("select");
      opts.select.forEach((optVal) => {
        const o = document.createElement("option");
        o.value = optVal;
        o.textContent = optVal;
        if (optVal === value) o.selected = true;
        input.appendChild(o);
      });
    } else if (opts.textarea) {
      input = document.createElement("textarea");
      input.rows = opts.rows || 3;
      input.value = value || "";
    } else {
      input = document.createElement("input");
      input.type = "text";
      input.value = value || "";
    }
    input.addEventListener("blur", () => {
      const newVal = input.value.trim();
      if (newVal && newVal !== (value || "")) onCommit(newVal);
    });
    wrap.appendChild(input);
    return wrap;
  }

  function listEditor(labelText, items, onAdd, onRemove) {
    const wrap = document.createElement("div");
    wrap.className = "jm-field jm-list-field";
    const label = document.createElement("span");
    label.className = "jm-field-label";
    label.textContent = labelText;
    wrap.appendChild(label);

    (items || []).forEach((item) => {
      const row = document.createElement("div");
      row.className = "jm-list-item";
      const text = document.createElement("span");
      text.textContent = item;
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "jm-list-remove";
      removeBtn.textContent = "✕";
      removeBtn.addEventListener("click", () => onRemove(item));
      row.appendChild(text);
      row.appendChild(removeBtn);
      wrap.appendChild(row);
    });

    const addRow = document.createElement("div");
    addRow.className = "jm-list-add";
    const addInput = document.createElement("input");
    addInput.type = "text";
    addInput.placeholder = `Add ${labelText.toLowerCase()}…`;
    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "neo-btn neo-btn-small neo-btn-yellow";
    addBtn.textContent = "Add";
    const submitAdd = () => {
      const val = addInput.value.trim();
      if (!val) return;
      addInput.value = "";
      onAdd(val);
    };
    addBtn.addEventListener("click", submitAdd);
    addInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        submitAdd();
      }
    });
    addRow.appendChild(addInput);
    addRow.appendChild(addBtn);
    wrap.appendChild(addRow);
    return wrap;
  }

  function sectionTitle(text) {
    const h = document.createElement("h4");
    h.className = "jm-section-title";
    h.textContent = text;
    return h;
  }

  function renderDraftForm(data) {
    const jobState = data.job_state || {};
    const jdVersions = data.jd_versions;
    const hasJd = jdVersions && (jdVersions["1"] || jdVersions["2"]);

    if (!jobState.job_title && !hasJd) {
      draftEmpty.hidden = false;
      draftForm.hidden = true;
      return;
    }
    draftEmpty.hidden = true;
    draftForm.hidden = false;
    clearChildren(draftForm);

    if (data.selected_version) viewVersion = data.selected_version;
    else if (jdVersions && !jdVersions[viewVersion]) viewVersion = Object.keys(jdVersions)[0] || "1";

    const header = document.createElement("div");
    header.className = "jm-draft-header";
    const headerTitle = document.createElement("h3");
    headerTitle.textContent = "Review & Edit Draft";
    header.appendChild(headerTitle);
    if (hasJd && jdVersions["1"] && jdVersions["2"]) {
      const toggle = document.createElement("div");
      toggle.className = "jm-version-toggle";
      ["1", "2"].forEach((v) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = `Version ${v}`;
        // Reflects the backend's CONFIRMED selection, not just which version is being viewed —
        // right after generation, selected_version is null (nothing confirmed yet, Publish
        // stays disabled), so neither button shows active until one is actually clicked. A
        // click always sends the confirmation, even for the version already on screen, since
        // "being displayed" and "confirmed to the backend" aren't the same thing yet at that
        // point — a no-op guard here previously left Publish permanently disabled.
        btn.className = v === data.selected_version ? "active" : "";
        btn.addEventListener("click", () => {
          viewVersion = v;
          sendMessage(`I prefer ${v}.`);
        });
        toggle.appendChild(btn);
      });
      header.appendChild(toggle);
    } else {
      const badge = document.createElement("span");
      badge.className = "neo-pill";
      badge.textContent = "AI GENERATED";
      header.appendChild(badge);
    }
    draftForm.appendChild(header);

    const jd = (jdVersions && jdVersions[viewVersion]) || {};

    draftForm.appendChild(
      fieldRow("Job Title", jobState.job_title, (v) => sendMessage(`Change the job title to ${v}.`))
    );

    draftForm.appendChild(
      fieldRow(
        "Job Description",
        jd.job_summary || "",
        (v) => sendMessage(`Please refine the job description: update the summary to read: "${v}"`),
        { textarea: true, rows: 4 }
      )
    );

    const grid1 = document.createElement("div");
    grid1.className = "jm-field-grid";
    grid1.appendChild(
      fieldRow("Experience Level", jobState.experience, (v) => sendMessage(`Set the experience level to ${v}.`))
    );
    grid1.appendChild(
      fieldRow(
        "Employment Type",
        jobState.employment_type,
        (v) => sendMessage(`Set the employment type to ${v}.`),
        { select: ["Full-time", "Part-time", "Contract", "Internship"] }
      )
    );
    draftForm.appendChild(grid1);

    const grid2 = document.createElement("div");
    grid2.className = "jm-field-grid";
    grid2.appendChild(fieldRow("Salary Range", jobState.salary, (v) => sendMessage(`Set the salary range to ${v}.`)));
    grid2.appendChild(
      fieldRow(
        "Location",
        [jobState.location, jobState.work_mode].filter(Boolean).join(" · "),
        (v) => sendMessage(`Set the location to ${v}.`)
      )
    );
    draftForm.appendChild(grid2);

    draftForm.appendChild(
      fieldRow("Deadline", jobState.deadline, (v) => sendMessage(`Set the application deadline to ${v}.`))
    );

    draftForm.appendChild(
      listEditor(
        "Responsibilities",
        jobState.responsibilities,
        (v) => sendMessage(`Add "${v}" as a responsibility.`),
        (v) => sendMessage(`Remove "${v}" from the responsibilities.`)
      )
    );

    draftForm.appendChild(
      listEditor(
        "Required Skills",
        jobState.required_skills,
        (v) => sendMessage(`Add ${v} as a required skill.`),
        (v) => sendMessage(`Remove ${v} from the required skills.`)
      )
    );

    draftForm.appendChild(
      listEditor(
        "Preferred Skills",
        jobState.preferred_skills,
        (v) => sendMessage(`Add ${v} as a preferred skill.`),
        (v) => sendMessage(`Remove ${v} from the preferred skills.`)
      )
    );

    if (hasJd) {
      const details = document.createElement("details");
      details.className = "jm-expandable";
      const summary = document.createElement("summary");
      summary.textContent = "Full job description detail";
      details.appendChild(summary);
      const body = document.createElement("div");
      body.className = "jm-expandable-body";
      [
        ["About the Role", jd.about_role],
        ["Major Accountabilities", jd.major_accountabilities],
        ["Minimum Requirements", jd.minimum_requirements],
        ["Required Qualifications", jd.required_qualifications],
        ["Preferred Qualifications", jd.preferred_qualifications],
        ["Stand Out", jd.stand_out],
        ["Benefits", jd.benefits],
      ].forEach(([label, val]) => {
        if (!val || (Array.isArray(val) && val.length === 0)) return;
        body.appendChild(sectionTitle(label));
        if (Array.isArray(val)) {
          const ul = document.createElement("ul");
          val.forEach((item) => {
            const li = document.createElement("li");
            li.textContent = item;
            ul.appendChild(li);
          });
          body.appendChild(ul);
        } else {
          const p = document.createElement("p");
          p.textContent = val;
          body.appendChild(p);
        }
      });
      details.appendChild(body);
      draftForm.appendChild(details);
    }

    // Company context — read-only from the stored profile, with a per-job override affordance.
    const overrides = jobState.company_overrides || {};
    const companyBox = document.createElement("div");
    companyBox.className = "jm-company-context";
    companyBox.appendChild(sectionTitle("Company Context"));
    [
      ["company_overview", "Overview"],
      ["company_culture", "Culture"],
      ["benefits", "Benefits"],
      ["why_join_us", "Why Join Us"],
    ].forEach(([key, label]) => {
      const current = overrides[key] || (companyProfileCache && companyProfileCache[key]) || "";
      companyBox.appendChild(
        fieldRow(
          `${label} (this job only)`,
          current,
          (v) =>
            sendMessage(
              `For this job specifically, use this ${label.toLowerCase()} instead of the default company profile: "${v}"`
            ),
          { textarea: true, rows: 2 }
        )
      );
    });
    draftForm.appendChild(companyBox);

    const actions = document.createElement("div");
    actions.className = "jm-draft-actions";

    const regenBtn = document.createElement("button");
    regenBtn.type = "button";
    regenBtn.className = "neo-btn";
    regenBtn.textContent = "⟳ Regenerate with AI";
    regenBtn.addEventListener("click", () => sendMessage("Please regenerate the job description from scratch."));
    actions.appendChild(regenBtn);

    const saveDraftBtn = document.createElement("button");
    saveDraftBtn.type = "button";
    saveDraftBtn.className = "neo-btn";
    saveDraftBtn.textContent = "Save as Draft";
    saveDraftBtn.addEventListener("click", () => closeModal());
    actions.appendChild(saveDraftBtn);

    const publishBtn = document.createElement("button");
    publishBtn.type = "button";
    publishBtn.className = "neo-btn neo-btn-solid";
    const alreadyPublished = data.job_record && data.job_record.status === "published" && data.phase === "published";
    publishBtn.textContent = alreadyPublished ? "Published ✓" : data.phase === "editing" ? "Publish Edit →" : "Publish Job →";
    publishBtn.disabled = !hasJd || !data.selected_version || data.jd_stale || alreadyPublished;
    publishBtn.addEventListener("click", () => {
      sendMessage(data.phase === "editing" ? "Please publish this edit." : "Yes, please publish this job now.");
    });
    actions.appendChild(publishBtn);

    draftForm.appendChild(actions);

    if (alreadyPublished && data.job_record) {
      const posted = document.createElement("div");
      posted.className = "jm-published-note";
      posted.innerHTML = "";
      const strong = document.createElement("strong");
      strong.textContent = `✓ Published as ${data.job_record.job_id}`;
      posted.appendChild(strong);
      draftForm.appendChild(posted);
    }
  }

  function renderAttachmentChip() {
    clearChildren(attachmentSlot);
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
    remove.addEventListener("click", () => {
      pendingFile = null;
      fileInput.value = "";
      renderAttachmentChip();
    });
    chip.appendChild(remove);
    attachmentSlot.appendChild(chip);
  }

  attachBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    pendingFile = file;
    renderAttachmentChip();
  });

  function updateStatusBar(data) {
    const labels = {
      collecting: "drafting mode • auto-save",
      summary: "drafting mode • auto-save",
      jd_selection: "reviewing draft • auto-save",
      publish_confirm: "ready to publish",
      published: "published",
      editing: "editing published job • unsaved changes",
    };
    statusEl.textContent = labels[data.phase] || "drafting mode • auto-save";
    progressEl.textContent = data.job_state && data.job_state.job_title ? data.job_state.job_title : "New job";
  }

  async function sendMessage(text) {
    clearError();
    const file = pendingFile;
    const displayText = file ? (text ? `📎 ${file.name}\n${text}` : `📎 ${file.name}`) : text;
    appendMessage("user", displayText);
    chatInput.value = "";
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
      currentData = data;
      currentPhase = data.phase;
      if (data.phase === "published") {
        localStorage.removeItem(STORAGE_KEY);
        if (window.showToast) window.showToast(`Job ${data.job_record ? data.job_record.job_id : ""} published successfully.`, "success");
        window.dispatchEvent(new CustomEvent("jobmodal:published"));
      } else {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ session_id: sessionId, job_title: data.job_state && data.job_state.job_title, updated_at: new Date().toISOString() })
        );
      }
      renderMessages(data.messages, data.suggested_options);
      renderDraftForm(data);
      updateStatusBar(data);
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

  let companyProfileCache = null;

  function openModal() {
    overlay.hidden = false;
    document.body.style.overflow = "hidden";
    setTimeout(() => chatInput.focus(), 50);
  }

  function closeModal() {
    overlay.hidden = true;
    document.body.style.overflow = "";
    if (window.loadJobsFromModal) window.loadJobsFromModal();
  }

  closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !overlay.hidden) closeModal();
  });

  window.addEventListener("jobmodal:published", () => {
    if (typeof loadJobs === "function") loadJobs();
  });

  async function startFresh() {
    sessionId = null;
    currentPhase = null;
    pendingFile = null;
    currentData = null;
    viewVersion = "1";
    renderAttachmentChip();
    clearError();
    renderMessages([], []);
    draftEmpty.hidden = false;
    draftForm.hidden = true;
    statusEl.textContent = "drafting mode • auto-save";
    progressEl.textContent = "New job";
  }

  async function openExisting(existingSessionId) {
    sessionId = existingSessionId;
    try {
      const data = await api.getChat(sessionId);
      currentData = data;
      currentPhase = data.phase;
      renderMessages(data.messages, data.suggested_options);
      renderDraftForm(data);
      updateStatusBar(data);
    } catch (err) {
      showToast(err.message || "Couldn't load that job.", "error");
      startFresh();
    }
  }

  window.openJobModal = async function (existingSessionId) {
    if (!companyProfileCache) {
      try {
        companyProfileCache = await api.getCompanyProfile();
      } catch (_) {
        companyProfileCache = {};
      }
    }
    if (existingSessionId) {
      await openExisting(existingSessionId);
    } else {
      await startFresh();
    }
    openModal();
  };

  window.loadJobsFromModal = function () {
    if (typeof loadJobs === "function") loadJobs();
  };
})();
