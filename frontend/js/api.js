const API_BASE = "/api";

async function apiRequest(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (err) {
    throw new Error("Network error — could not reach the server. Please check your connection and try again.");
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) {
      // ignore parse failure, use default detail
    }
    const err = new Error(detail);
    err.status = response.status;
    throw err;
  }

  if (response.status === 204) return null;
  return response.json();
}

// Reads a chat-turn SSE stream (both /api/chat and /api/chat/upload use this same event shape):
// "status" events carry a real, backend-driven processing label (never fabricated — see
// stream_graph_turn in chat.py), and a single terminal "result" event carries the same
// ChatResponse shape the endpoints have always returned. onStatus is called for every status
// event so the caller can show live progress; the resolved value is the result payload.
async function streamChatTurn(url, fetchOptions, onStatus) {
  let response;
  try {
    response = await fetch(url, fetchOptions);
  } catch (err) {
    throw new Error("Network error — could not reach the server. Please check your connection and try again.");
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) {
      // ignore parse failure, use default detail
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data: "));
      if (!dataLine) continue;
      const event = JSON.parse(dataLine.slice(6));
      if (event.type === "status") {
        if (onStatus) onStatus(event.label);
      } else if (event.type === "error") {
        throw new Error(event.detail || "Something went wrong. Please try again.");
      } else if (event.type === "result") {
        delete event.type;
        result = event;
      }
    }
  }

  if (!result) throw new Error("The connection ended unexpectedly. Please try again.");
  return result;
}

const api = {
  signup: (payload) => apiRequest("/auth/signup", { method: "POST", body: JSON.stringify(payload) }),
  signin: (email, password) =>
    apiRequest("/auth/signin", { method: "POST", body: JSON.stringify({ email, password }) }),
  logout: () => apiRequest("/auth/logout", { method: "POST" }),
  getMe: () => apiRequest("/auth/me"),

  postChat: (sessionId, message, onStatus) =>
    streamChatTurn(
      `${API_BASE}/chat`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message }),
      },
      onStatus
    ),
  postChatUpload: (sessionId, message, file, onStatus) => {
    const formData = new FormData();
    if (sessionId) formData.append("session_id", sessionId);
    formData.append("message", message);
    formData.append("file", file);
    return streamChatTurn(`${API_BASE}/chat/upload`, { method: "POST", body: formData }, onStatus);
  },
  getChat: (sessionId) => apiRequest(`/chat/${encodeURIComponent(sessionId)}`),

  getCompanyProfile: () => apiRequest("/company-profile"),
  putCompanyProfile: (updates) =>
    apiRequest("/company-profile", { method: "PUT", body: JSON.stringify(updates) }),

  getJobs: () => apiRequest("/jobs"),
  setJobAcceptingApplications: (sessionId, accepting) =>
    apiRequest(`/jobs/${encodeURIComponent(sessionId)}/accepting-applications`, {
      method: "PUT",
      body: JSON.stringify({ accepting_applications: accepting }),
    }),
  deleteJob: (sessionId) => apiRequest(`/jobs/${encodeURIComponent(sessionId)}`, { method: "DELETE" }),

  getPublicJobs: () => apiRequest("/public/jobs"),
  getPublicJob: (jobId) => apiRequest(`/public/jobs/${encodeURIComponent(jobId)}`),

  getAdminJobs: () => apiRequest("/admin/jobs"),
  getAdminJob: (id) => apiRequest(`/admin/jobs/${encodeURIComponent(id)}`),
  getAdminCompany: (id) => apiRequest(`/admin/companies/${encodeURIComponent(id)}`),
};
