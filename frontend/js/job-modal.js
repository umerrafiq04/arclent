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

  // Single-select: tapping a chip sends immediately (e.g. work mode, experience band, job
  // title). Multi-select: tapping toggles the chip on/off, nothing sends until "Add Selected" —
  // for questions like "any other required skills?" where picking several at once (Python + SQL
  // + React) is the whole point. The bot decides which mode via options_multi_select per turn.
  function appendChips(afterRow, options, multiSelect) {
    if (!options || options.length === 0) return;
    const row = document.createElement("div");
    row.className = "jm-chip-row";

    if (!multiSelect) {
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
    } else {
      const selected = new Set();
      const confirmBtn = document.createElement("button");
      confirmBtn.type = "button";
      confirmBtn.className = "jm-chip-confirm";
      confirmBtn.textContent = "Add Selected";
      confirmBtn.disabled = true;
      confirmBtn.addEventListener("click", () => {
        row.querySelectorAll(".jm-chip, .jm-chip-confirm").forEach((c) => (c.disabled = true));
        sendMessage(Array.from(selected).join(", "));
      });

      options.forEach((opt) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "jm-chip jm-chip-toggle";
        chip.textContent = opt;
        chip.addEventListener("click", () => {
          if (selected.has(opt)) {
            selected.delete(opt);
            chip.classList.remove("selected");
          } else {
            selected.add(opt);
            chip.classList.add("selected");
          }
          confirmBtn.disabled = selected.size === 0;
          confirmBtn.textContent = selected.size ? `Add Selected (${selected.size})` : "Add Selected";
        });
        row.appendChild(chip);
      });
      row.appendChild(confirmBtn);
    }

    afterRow.insertAdjacentElement("afterend", row);
    messageList.scrollTop = messageList.scrollHeight;
    return row;
  }

  // The backend only ever sets asking_about_field when its latest reply is a live question
  // about one specific OPTIONAL field (see apply_updates in nodes.py) — never fabricated
  // client-side. Recruiters who don't have that piece of info (or just don't want to answer)
  // need a way out of the question besides typing free text or hunting for the right chip.
  function appendSkipButton(afterRow, field) {
    const row = document.createElement("div");
    row.className = "jm-skip-row";
    const skipBtn = document.createElement("button");
    skipBtn.type = "button";
    skipBtn.className = "jm-skip-btn";
    skipBtn.textContent = "Skip this";
    skipBtn.addEventListener("click", () => skipCurrentField(skipBtn));
    row.appendChild(skipBtn);
    afterRow.insertAdjacentElement("afterend", row);
    messageList.scrollTop = messageList.scrollHeight;
  }

  // Skipping ONLY ever happens here — a direct API call, never a chat message. There's nothing
  // for the recruiter to have "said", so unlike sendMessage this adds no user bubble to the
  // transcript at all — just the bot's next question appearing.
  async function skipCurrentField(button) {
    clearError();
    button.disabled = true;
    try {
      const data = await api.skipField(sessionId);
      currentData = data;
      currentPhase = data.phase;
      renderMessages(data.messages, data.suggested_options, data.options_multi_select, data.asking_about_field);
      renderDraftForm(data);
      updateStatusBar(data);
    } catch (err) {
      showError(err.message || "Couldn't skip this. Please try again.");
      button.disabled = false;
    }
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

  // A wide pool spanning different departments — four are picked at random each time a fresh
  // job modal opens (see pickRoleChips below) so the opening suggestions don't feel like the
  // same static four options every single time.
  const ROLE_CHIP_POOL = [
    "Software Engineer", "Data Analyst", "Product Manager", "Sales Executive",
    "Marketing Manager", "UX Designer", "Customer Support Specialist", "Operations Manager",
    "HR Business Partner", "Financial Analyst", "DevOps Engineer", "Content Writer",
    "Business Development Manager", "QA Engineer", "Recruiter", "Account Executive",
  ];

  function pickRoleChips(count) {
    const pool = ROLE_CHIP_POOL.slice();
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, count);
  }

  function renderMessages(messages, suggestedOptions, multiSelect, askingAboutField) {
    clearChildren(messageList);
    if (!messages || messages.length === 0) {
      // Greeting is static text (no analyze_turn call has happened yet), but it's still a real
      // question — it gets the same chip treatment as every other AI question, not just plain
      // centered text with nothing tappable under it. Always single-select (one role to start).
      // window.recruiterFirstName is set by recruiter.js once /auth/me resolves — falls back to
      // the generic greeting on the rare chance the modal opens before that request lands.
      const greeting = window.recruiterFirstName
        ? `Hi ${window.recruiterFirstName}, what are you hiring for today? 👋`
        : "What are you hiring for today? 👋";
      const row = appendMessage("ai", greeting);
      appendChips(row, pickRoleChips(4), false);
      return;
    }
    let lastAiRow = null;
    messages.forEach((m) => {
      if (m.jd_document) return; // JD content lives in the draft pane, not the chat, here
      const row = appendMessage(m.role, m.content);
      if (m.role !== "user") lastAiRow = row;
    });
    if (!lastAiRow) return;
    let insertAfter = lastAiRow;
    if (suggestedOptions && suggestedOptions.length) {
      insertAfter = appendChips(lastAiRow, suggestedOptions, multiSelect) || lastAiRow;
    }
    if (askingAboutField) {
      appendSkipButton(insertAfter, askingAboutField);
    }
  }

  // Direct, silent job_state edit — no chat message, no processing status, no bot reply. This
  // is what every draft-form field commit calls instead of sendMessage.
  async function patchField(patch) {
    try {
      const data = await api.patchJobState(sessionId, patch);
      currentData = data;
      currentPhase = data.phase;
      renderDraftForm(data);
      updateStatusBar(data);
    } catch (err) {
      showError(err.message || "Couldn't save that change. Please try again.");
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
      if (!value) {
        // Without an explicit placeholder, an unset field would default to displaying its
        // first real option as if it were already answered (a real browser <select> behavior,
        // not a bug in the data) — misleading for exactly the kind of field this app needs to
        // get right (e.g. Work Mode silently "looking like" Remote when nothing was said yet).
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Not set";
        placeholder.disabled = true;
        placeholder.selected = true;
        input.appendChild(placeholder);
      }
      opts.select.forEach((optVal) => {
        const o = document.createElement("option");
        o.value = optVal;
        o.textContent = optVal;
        if (optVal === value) o.selected = true;
        input.appendChild(o);
      });
      // Selects have no natural "blur to commit" moment the way typed fields do — commit on
      // change instead, immediately.
      input.addEventListener("change", () => onCommit(input.value));
    } else {
      if (opts.textarea) {
        input = document.createElement("textarea");
        input.rows = opts.rows || 3;
      } else {
        input = document.createElement("input");
        input.type = "text";
      }
      input.value = value || "";
      input.addEventListener("blur", () => {
        const newVal = input.value.trim();
        if (newVal !== (value || "")) onCommit(newVal);
      });
    }
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
    const hasJd = Boolean(jdVersions && jdVersions["1"]);

    if (!jobState.job_title && !hasJd) {
      draftEmpty.hidden = false;
      draftForm.hidden = true;
      return;
    }
    draftEmpty.hidden = true;
    draftForm.hidden = false;
    clearChildren(draftForm);

    const header = document.createElement("div");
    header.className = "jm-draft-header";
    const headerTitle = document.createElement("h3");
    headerTitle.textContent = "Review & Edit Draft";
    header.appendChild(headerTitle);
    if (hasJd) {
      const badge = document.createElement("span");
      badge.className = "neo-pill";
      badge.textContent = "AI GENERATED";
      header.appendChild(badge);
    }
    // Job Requisition ID is assigned at publish time — shows here (read-only, matches exactly
    // what the public job page and "Job Requisition ID" field display) once it exists, so
    // there's no gap between what the recruiter previews and what's actually live.
    if (data.job_record && data.job_record.job_id) {
      const reqIdBadge = document.createElement("span");
      reqIdBadge.className = "neo-pill";
      reqIdBadge.textContent = `Req ID: ${data.job_record.job_id}`;
      header.appendChild(reqIdBadge);
    }
    draftForm.appendChild(header);

    const jd = (jdVersions && jdVersions["1"]) || {};

    draftForm.appendChild(
      fieldRow("Job Title", jobState.job_title, (v) => patchField({ field_updates: { job_title: v } }))
    );
    draftForm.appendChild(
      fieldRow("Job Category", jobState.job_category, (v) => patchField({ field_updates: { job_category: v } }))
    );

    if (hasJd) {
      draftForm.appendChild(
        fieldRow(
          "Job Description",
          jd.job_summary || "",
          (v) => patchField({ jd_text_updates: { job_summary: v } }),
          { textarea: true, rows: 4 }
        )
      );
    }

    const grid1 = document.createElement("div");
    grid1.className = "jm-field-grid";
    grid1.appendChild(
      fieldRow("Experience Level", jobState.experience, (v) => patchField({ field_updates: { experience: v } }))
    );
    grid1.appendChild(
      fieldRow(
        "Employment Type",
        jobState.employment_type,
        (v) => patchField({ field_updates: { employment_type: v } }),
        { select: ["Full-time", "Part-time", "Contract", "Internship"] }
      )
    );
    draftForm.appendChild(grid1);

    // Location and Work Mode are deliberately separate fields — Location is an actual place
    // (a city, or "Worldwide" for fully remote), Work Mode is the arrangement. Combining them
    // used to send an ambiguous "set the location to X" instruction that could clobber either.
    const grid2 = document.createElement("div");
    grid2.className = "jm-field-grid";
    grid2.appendChild(
      fieldRow("Location", jobState.location, (v) => patchField({ field_updates: { location: v } }))
    );
    grid2.appendChild(
      fieldRow(
        "Work Mode",
        jobState.work_mode,
        (v) => patchField({ field_updates: { work_mode: v } }),
        { select: ["Remote", "Hybrid", "Onsite"] }
      )
    );
    draftForm.appendChild(grid2);

    const grid3 = document.createElement("div");
    grid3.className = "jm-field-grid";
    grid3.appendChild(
      fieldRow("Salary Range", jobState.salary, (v) => patchField({ field_updates: { salary: v } }))
    );
    grid3.appendChild(
      fieldRow("Deadline", jobState.deadline, (v) => patchField({ field_updates: { deadline: v } }))
    );
    draftForm.appendChild(grid3);

    const grid4 = document.createElement("div");
    grid4.className = "jm-field-grid";
    grid4.appendChild(
      fieldRow(
        "Education",
        jobState.education,
        (v) => patchField({ field_updates: { education: v } }),
        { select: ["Bachelor's degree", "Master's degree", "Not required"] }
      )
    );
    draftForm.appendChild(grid4);

    draftForm.appendChild(
      listEditor(
        "Responsibilities",
        jobState.responsibilities,
        (v) => patchField({ list_operations: [{ field: "responsibilities", operation: "ADD", values: [v] }] }),
        (v) => patchField({ list_operations: [{ field: "responsibilities", operation: "REMOVE", values: [v] }] })
      )
    );

    draftForm.appendChild(
      listEditor(
        "Required Skills",
        jobState.required_skills,
        (v) => patchField({ list_operations: [{ field: "required_skills", operation: "ADD", values: [v] }] }),
        (v) => patchField({ list_operations: [{ field: "required_skills", operation: "REMOVE", values: [v] }] })
      )
    );

    draftForm.appendChild(
      listEditor(
        "Preferred Skills",
        jobState.preferred_skills,
        (v) => patchField({ list_operations: [{ field: "preferred_skills", operation: "ADD", values: [v] }] }),
        (v) => patchField({ list_operations: [{ field: "preferred_skills", operation: "REMOVE", values: [v] }] })
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
        ["Company Overview", jd.company_overview],
        ["Why Join", jd.why_company],
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
          (v) => patchField({ company_overrides: { [key]: v } }),
          { textarea: true, rows: 2 }
        )
      );
    });
    draftForm.appendChild(companyBox);

    const actions = document.createElement("div");
    actions.className = "jm-draft-actions";

    const generateBtn = document.createElement("button");
    generateBtn.type = "button";
    generateBtn.className = "neo-btn";
    generateBtn.textContent = hasJd ? "⟳ Regenerate" : "✦ Generate Full Description";
    // Deterministic sufficiency signal from the backend (hard floor + the standard checklist
    // resolved) — never inferred client-side. Applies uniformly to Generate AND Regenerate: if a
    // checklist field gets cleared via the draft panel after a JD already exists, Regenerate
    // correctly re-disables until it's answered again, same principle either way.
    generateBtn.disabled = !data.ready_to_generate;
    generateBtn.title = data.ready_to_generate ? "" : "Finish the guided questions in chat to enable this.";
    generateBtn.addEventListener("click", () => generateNow(generateBtn, hasJd));
    actions.appendChild(generateBtn);

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
    publishBtn.addEventListener("click", () => publishNow(publishBtn));
    actions.appendChild(publishBtn);

    draftForm.appendChild(actions);

    if (alreadyPublished && data.job_record) {
      const posted = document.createElement("div");
      posted.className = "jm-published-note";
      const strong = document.createElement("strong");
      strong.textContent = `✓ Published as ${data.job_record.job_id}`;
      posted.appendChild(strong);
      draftForm.appendChild(posted);
    }
  }

  // Generation ONLY ever happens here — a direct API call the button makes, never a side effect
  // of a chat message (see the backend's REQUEST_JD_GENERATION guidance: chat can acknowledge
  // and point here, but never generates itself). No fake "please generate" user bubble is added
  // — just a processing indicator while the call is in flight, then the result.
  async function generateNow(button, isRegenerate) {
    clearError();
    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = isRegenerate ? "Regenerating…" : "Generating…";
    showProcessingStatus(isRegenerate ? "Regenerating your job description..." : "Creating your job description...");
    try {
      const data = await api.generateJd(sessionId);
      hideProcessingStatus();
      currentData = data;
      currentPhase = data.phase;
      renderMessages(data.messages, data.suggested_options, data.options_multi_select, data.asking_about_field);
      renderDraftForm(data);
      updateStatusBar(data);
    } catch (err) {
      hideProcessingStatus();
      showError(err.message || "Couldn't generate the job description. Please try again.");
      button.disabled = false;
      button.textContent = originalText;
    }
  }

  // Publishing ONLY ever happens here — a direct API call the button makes, never a side effect
  // of a chat message (see the backend's CONFIRM_PUBLISH guidance: chat can acknowledge, but
  // only this endpoint actually flips the job live). The modal closes itself right after a
  // successful publish — leaving it open invited the recruiter to keep chatting against an
  // already-published job, which the bot could misread as a request to touch it again right
  // away ("now" -> "Generating the updated job description..."). Further changes belong to a
  // deliberate later "Manage -> Edit" reopen, not this same open window.
  async function publishNow(button) {
    clearError();
    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = "Publishing…";
    try {
      const data = await api.publishJob(sessionId);
      currentData = data;
      currentPhase = data.phase;
      localStorage.removeItem(STORAGE_KEY);
      renderMessages(data.messages, data.suggested_options, data.options_multi_select, data.asking_about_field);
      renderDraftForm(data);
      updateStatusBar(data);
      celebrate();
      if (window.showToast) {
        window.showToast(`Job ${data.job_record ? data.job_record.job_id : ""} published successfully.`, "success");
      }
      window.dispatchEvent(new CustomEvent("jobmodal:published"));
      setTimeout(closeModal, 1400);
    } catch (err) {
      showError(err.message || "Couldn't publish this job. Please try again.");
      button.disabled = false;
      button.textContent = originalText;
    }
  }

  // Lightweight, dependency-free confetti burst — no external library, just a handful of
  // absolutely-positioned divs animated with the Web Animations API, matching the neo palette.
  function celebrate() {
    const colors = ["#ffd23f", "#3ddc84", "#14140f", "#ffffff"];
    const container = document.createElement("div");
    container.className = "jm-confetti";
    const originX = window.innerWidth / 2;
    const originY = window.innerHeight / 3;
    for (let i = 0; i < 60; i++) {
      const piece = document.createElement("span");
      piece.className = "jm-confetti-piece";
      piece.style.background = colors[i % colors.length];
      piece.style.left = `${originX}px`;
      piece.style.top = `${originY}px`;
      const angle = Math.random() * Math.PI * 2;
      const distance = 160 + Math.random() * 220;
      const dx = Math.cos(angle) * distance;
      const dy = Math.sin(angle) * distance + 120;
      const rotate = Math.random() * 720 - 360;
      container.appendChild(piece);
      piece.animate(
        [
          { transform: "translate(0, 0) rotate(0deg)", opacity: 1 },
          { transform: `translate(${dx}px, ${dy}px) rotate(${rotate}deg)`, opacity: 0 },
        ],
        { duration: 900 + Math.random() * 500, easing: "cubic-bezier(0.2, 0.8, 0.3, 1)" }
      );
    }
    document.body.appendChild(container);
    setTimeout(() => container.remove(), 1500);
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
      if (data.phase !== "published") {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ session_id: sessionId, job_title: data.job_state && data.job_state.job_title, updated_at: new Date().toISOString() })
        );
      }
      renderMessages(data.messages, data.suggested_options, data.options_multi_select, data.asking_about_field);
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
  let closeTimer = null;

  // Smooth open/close: add .open a frame after unhiding (so the initial state paints first and
  // the transition actually runs), and on close, wait for the transition to finish before
  // setting hidden — an iOS-sheet-like settle instead of an instant show/hide.
  function openModal() {
    if (closeTimer) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }
    overlay.hidden = false;
    document.body.style.overflow = "hidden";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => overlay.classList.add("open"));
    });
    setTimeout(() => chatInput.focus(), 340);
  }

  function closeModal() {
    overlay.classList.remove("open");
    document.body.style.overflow = "";
    closeTimer = setTimeout(() => {
      overlay.hidden = true;
      closeTimer = null;
    }, 360);
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
    renderAttachmentChip();
    clearError();
    renderMessages([], [], false);
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
      renderMessages(data.messages, data.suggested_options, data.options_multi_select, data.asking_about_field);
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
