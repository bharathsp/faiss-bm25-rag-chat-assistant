const state = {
  sessionId: null,
  lastAnswer: "",
  isRecording: false,
  isStreaming: false,
  mediaRecorder: null,
  audioChunks: [],
};

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadList = document.getElementById("uploadList");
const uploadProgress = document.getElementById("uploadProgress");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");
const suggestionsEl = document.getElementById("suggestions");
const docCount = document.getElementById("docCount");
const statusPill = document.getElementById("statusPill");
const themeToggle = document.getElementById("themeToggle");
const clearBtn = document.getElementById("clearBtn");
const micBtn = document.getElementById("micBtn");
const speakBtn = document.getElementById("speakBtn");
const sendBtn = chatForm.querySelector(".send-btn");
const toast = document.getElementById("toast");

if (window.marked) {
  marked.setOptions({ breaks: true, gfm: true });
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 2800);
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  themeToggle.textContent = theme === "dark" ? "🌙" : "☀️";
}

function initTheme() {
  const saved = localStorage.getItem("theme") || "dark";
  setTheme(saved);
}

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  setTheme(current === "dark" ? "light" : "dark");
});

function renderMarkdown(text) {
  if (window.marked) {
    return marked.parse(text || "");
  }
  return (text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

function markdownToPlainText(text) {
  if (!text) return "";

  if (window.marked) {
    const div = document.createElement("div");
    div.innerHTML = marked.parse(text);
    return (div.innerText || div.textContent || "").trim();
  }

  return text
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/```\w*\n?/g, "").replace(/```/g, ""))
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/_(.+?)_/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/~~(.+?)~~/g, "$1")
    .trim();
}

function scrollChatToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderSuggestions(items = []) {
  suggestionsEl.innerHTML = "";
  items.forEach((text) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      if (state.isStreaming) return;
      chatInput.value = text;
      chatForm.requestSubmit();
    });
    suggestionsEl.appendChild(chip);
  });
}

function createMessageShell(role) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  return { wrap, bubble };
}

function appendMessage(role, content, sources = []) {
  const { bubble } = createMessageShell(role);
  const body = document.createElement("div");
  body.className = "md-content";

  if (role === "assistant") {
    body.innerHTML = renderMarkdown(content);
  } else {
    body.textContent = content;
  }

  bubble.appendChild(body);

  if (sources.length) {
    appendSources(bubble, sources);
  }

  scrollChatToBottom();
}

function appendSources(bubble, sources) {
  const src = document.createElement("div");
  src.className = "sources";
  src.textContent = `Sources: ${sources.join(", ")}`;
  bubble.appendChild(src);
}

function createStreamingMessage() {
  const { bubble } = createMessageShell("assistant");
  const body = document.createElement("div");
  body.className = "md-content";
  body.innerHTML = '<span class="stream-cursor">▋</span>';
  bubble.appendChild(body);
  scrollChatToBottom();
  return { bubble, body };
}

function updateStreamingMessage(body, text, streaming = true) {
  body.innerHTML = renderMarkdown(text);
  if (streaming) {
    body.insertAdjacentHTML("beforeend", '<span class="stream-cursor">▋</span>');
  }
  scrollChatToBottom();
}

function setChatBusy(busy) {
  state.isStreaming = busy;
  chatInput.disabled = busy;
  sendBtn.disabled = busy;
  if (busy) {
    sendBtn.textContent = "Streaming...";
  } else {
    sendBtn.textContent = "Send";
  }
}

function updateStatus(data) {
  docCount.textContent = `${data.chunk_count || 0} chunks`;
  statusPill.textContent = data.ready ? "RAG ready" : "Waiting for docs";
  statusPill.classList.toggle("ready", !!data.ready);
  renderDocuments(data.documents || []);
}

function renderDocuments(documents) {
  uploadList.innerHTML = "";

  if (!documents.length) {
    const empty = document.createElement("div");
    empty.className = "upload-list-empty";
    empty.textContent = "No documents indexed yet";
    uploadList.appendChild(empty);
    return;
  }

  documents.forEach((doc) => {
    const item = document.createElement("div");
    item.className = "doc-item";

    const info = document.createElement("div");
    info.className = "doc-info";
    info.innerHTML = `
      <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
      <span class="doc-meta">${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}</span>
    `;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "doc-remove";
    removeBtn.title = "Remove document";
    removeBtn.setAttribute("aria-label", `Remove ${doc.filename}`);
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeDocument(doc.filename);
    });

    item.appendChild(info);
    item.appendChild(removeBtn);
    uploadList.appendChild(item);
  });
}

async function removeDocument(filename) {
  if (!confirm(`Remove "${filename}" from the index?`)) return;

  try {
    const res = await fetch(`/api/documents?filename=${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Remove failed");

    updateStatus(data);
    renderSuggestions(data.suggestions || []);
    showToast(`Removed ${filename}`);
  } catch (err) {
    showToast(err.message);
  }
}

async function fetchStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  updateStatus(data);
  if (data.ready) {
    const suggestRes = await fetch("/api/suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, mode: "initial" }),
    });
    const suggestData = await suggestRes.json();
    renderSuggestions(suggestData.suggestions || []);
  }
}

async function uploadFiles(files) {
  if (!files.length) return;

  uploadProgress.classList.remove("hidden");

  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText || "Upload failed");

    updateStatus({
      ready: data.ready ?? true,
      chunk_count: data.chunk_count,
      documents: data.documents,
    });
    renderSuggestions(data.suggestions || []);
    showToast(`Indexed ${data.total_chunks} chunks from ${data.processed.length} file(s)`);
  } catch (err) {
    showToast(err.message);
  } finally {
    uploadProgress.classList.add("hidden");
    fileInput.value = "";
  }
}

async function streamChat(message) {
  const { bubble, body } = createStreamingMessage();
  let fullText = "";

  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: state.sessionId }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Chat failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      const line = event.trim();
      if (!line.startsWith("data: ")) continue;

      const data = JSON.parse(line.slice(6));
      if (data.type === "token") {
        fullText += data.content;
        updateStreamingMessage(body, fullText, true);
      } else if (data.type === "done") {
        state.sessionId = data.session_id;
        state.lastAnswer = fullText;
        updateStreamingMessage(body, fullText, false);
        if (data.sources?.length) {
          appendSources(bubble, data.sources);
        }
        renderSuggestions(data.suggestions || []);
      } else if (data.type === "error") {
        throw new Error(data.message || "Streaming failed");
      }
    }
  }

  if (!fullText) {
    updateStreamingMessage(body, "No response received.", false);
  }
}

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  uploadFiles(e.dataTransfer.files);
});

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => uploadFiles(e.target.files));

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message || state.isStreaming) return;

  appendMessage("user", message);
  chatInput.value = "";
  setChatBusy(true);

  try {
    await streamChat(message);
  } catch (err) {
    appendMessage("assistant", `**Error:** ${err.message}`);
  } finally {
    setChatBusy(false);
    chatInput.focus();
  }
});

clearBtn.addEventListener("click", async () => {
  if (!confirm("Clear all uploaded documents, vectors, and chat memory?")) return;
  await fetch("/api/clear", { method: "DELETE" });
  location.reload();
});

micBtn.addEventListener("click", async () => {
  if (state.isRecording) {
    state.mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioChunks = [];
    state.mediaRecorder = new MediaRecorder(stream);
    state.mediaRecorder.ondataavailable = (e) => state.audioChunks.push(e.data);
    state.mediaRecorder.onstop = async () => {
      micBtn.classList.remove("recording");
      state.isRecording = false;
      stream.getTracks().forEach((track) => track.stop());

      const blob = new Blob(state.audioChunks, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");

      showToast("Transcribing with Whisper...");
      const res = await fetch("/api/transcribe", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Transcription failed");
      chatInput.value = data.text;
      chatInput.focus();
    };

    state.mediaRecorder.start();
    state.isRecording = true;
    micBtn.classList.add("recording");
    showToast("Recording... click mic to stop");
  } catch {
    showToast("Microphone access denied");
  }
});

speakBtn.addEventListener("click", () => {
  if (!state.lastAnswer) {
    showToast("No answer to read yet");
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(markdownToPlainText(state.lastAnswer));
  utterance.rate = 1;
  window.speechSynthesis.speak(utterance);
});

initTheme();
fetchStatus();
renderSuggestions([
  "Upload documents to get started",
  "What file types are supported?",
  "How does hybrid RAG work here?",
]);
