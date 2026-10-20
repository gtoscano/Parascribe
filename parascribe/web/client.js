import {SentenceGate} from "/v2/test/streaming.js";
const $ = id => document.getElementById(id);
const state = {session: null, ws: null, mic: null, context: null, processor: null,
  pending: new Map(), results: new Map(), flushes: new Map(), samples: 0,
  busy: false, ending: false, cohere: false, gate: null, segment: null,
  delivery: "manual", previewAt: 0, names: {}};

function notice(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
  const lines = $("events").textContent.split("\n").filter(Boolean);
  lines.push(`${new Date().toLocaleTimeString()} ${message}`);
  $("events").textContent = lines.slice(-40).join("\n");
}

function update() {
  const active = state.session?.status === "active";
  const streaming = !!state.ws;
  const locked = state.busy || state.ending;
  const idle = !locked && !streaming && !state.pending.size;
  for (const id of ["create", "attach", "capture", "sessionId"]) $(id).disabled = !idle;
  for (const id of ["mode", "language", "file", "delivery", "pause", "sensitivity"]) $(id).disabled = !idle;
  $("upload").disabled = !active || !idle || !$("file").files.length;
  $("start").disabled = !active || !idle || !window.isSecureContext || !navigator.mediaDevices;
  $("commit").disabled = !state.mic || locked || state.pending.size >= 4;
  $("stop").disabled = !state.mic || locked;
  $("end").disabled = !active || locked;
  $("delete").disabled = !state.session?.capture_voiceprints || locked;
  $("refresh").disabled = !state.session || locked;
  $("download").disabled = !state.session || locked;
  $("deleteOnEnd").disabled = !active || locked;
  $("nameSpeaker").disabled = !state.session || !$("speakerSelect").value;
  $("language").options[0].disabled = $("mode").value !== "parakeet";
  $("activity").textContent = `${state.mic ? "Microphone on" : "Microphone off"} · ${(state.samples / 16000).toFixed(1)}s unsent · ${state.pending.size} pending sentences`;
  $("sessionInfo").textContent = state.session
    ? `Session ${state.session.session_id} · ${state.session.status} · voice signatures ${state.session.capture_voiceprints ? "enabled" : "disabled"}`
    : "No session selected.";
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body));
  return body;
}
function sessionPath(suffix = "") {
  return `/v2/sessions/${encodeURIComponent(state.session.session_id)}${suffix}`;
}
function settings() {
  const mode = $("mode").value, language = $("language").value;
  if (mode !== "parakeet" && !language) throw new Error("Select a language for Cohere or comparison mode.");
  if (mode !== "parakeet" && !state.cohere) throw new Error("Cohere weights are not configured on this server. Select Parakeet or configure Cohere first.");
  return {asr_mode: mode, language: language || null};
}
function jsonPost(body) {
  return {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
}
async function run(action) {
  if (state.busy) return;
  state.busy = true; update();
  try { await action(); } catch (error) { notice(error.message, true); }
  finally { state.busy = false; update(); }
}
function element(tag, text) {
  const node = document.createElement(tag); node.textContent = text; return node;
}

function speakerLabel(id) {
  if (id) return state.names[id] ? `${state.names[id]} · ${id}` : id;
  // Unmatched voices are held until a second sentence confirms a new speaker.
  return state.session?.capture_voiceprints ? "Speaker pending confirmation" : "Speaker unidentified / recognition off";
}
function speakerControls() {
  const selected = $("speakerSelect").value;
  const ids = [...new Set([...state.results.values()].map(result => result.speaker_id).filter(Boolean))];
  $("speakerSelect").replaceChildren();
  if (!ids.length) $("speakerSelect").append(new Option("No speakers yet", ""));
  for (const id of ids) $("speakerSelect").append(new Option(speakerLabel(id), id));
  if (ids.includes(selected)) $("speakerSelect").value = selected;
  if (document.activeElement !== $("speakerName")) $("speakerName").value = state.names[$("speakerSelect").value] || "";
  $("nameSpeaker").disabled = !ids.length;
}
function showDraft(text, info = "Provisional text — may change") {
  $("draft").hidden = false;
  $("draftSpeaker").textContent = state.session?.capture_voiceprints ? "Speaker pending final voice match" : "Speaker recognition off";
  $("draftText").textContent = text;
  $("draftInfo").textContent = info;
}

function showResult(result, observedMs = null) {
  const previous = state.results.get(result.utterance_id);
  if (previous?.transcription && !result.transcription) return;
  state.results.set(result.utterance_id, result);
  const card = document.createElement("article");
  card.dataset.utterance = result.utterance_id;
  const badge = element("strong", speakerLabel(result.speaker_id)); badge.className = "speaker";
  card.append(badge, element("small", ` · Sentence ${result.sequence}`));
  card.append(element("p", result.text));
  const similarity = result.speaker?.similarity ?? result.speaker_similarity;
  card.append(element("small", `${result.language || "und"} · ${Number(result.audio_seconds).toFixed(1)}s audio${similarity == null ? "" : ` · voice similarity ${similarity.toFixed(3)}`}${observedMs == null ? "" : ` · result in ${(observedMs / 1000).toFixed(2)}s`}`));
  if (result.transcription) {
    const wrap = element("div", ""); wrap.className = "table-wrap";
    const table = document.createElement("table");
    const head = document.createElement("tr");
    for (const title of ["Model", "Transcript", "Inference", "RTF ↓"]) head.append(element("th", title));
    table.append(head);
    for (const [engine, output] of Object.entries(result.transcription.outputs)) {
      const row = document.createElement("tr");
      for (const value of [`${engine} (${output.quantization})`, output.text || "[No text]", `${output.inference_s.toFixed(3)}s`, output.real_time_factor?.toFixed(3) ?? "—"]) row.append(element("td", value));
      table.append(row);
    }
    wrap.append(table); card.append(wrap);
  }
  const old = Array.from($("results").querySelectorAll("article")).find(node => node.dataset.utterance === result.utterance_id);
  if (old) old.replaceWith(card); else $("results").append(card);
  const live = document.createElement("article"); live.dataset.utterance = result.utterance_id;
  live.append(badge.cloneNode(true), element("p", result.text));
  const existing = [...$("liveResults").children].find(node => node.dataset.utterance === result.utterance_id);
  if (existing) existing.replaceWith(live); else $("liveResults").append(live);
  speakerControls();
  $("empty").hidden = true;
  for (const item of result.speaker?.relabeled || []) {
    const earlier = state.results.get(item.utterance_id);
    if (earlier) showResult({...earlier, speaker_id: item.speaker_id, speaker_similarity: item.speaker_similarity, speaker: undefined});
  }
}

async function loadSession(id, reset = false) {
  const session = await api(`/v2/sessions/${encodeURIComponent(id)}`);
  if (reset) {
    state.results.clear(); $("results").replaceChildren($("empty")); $("empty").hidden = false;
    $("liveResults").replaceChildren(); $("draft").hidden = true;
    try {
      const names = JSON.parse(localStorage.getItem(`parascribe-names-${id}`) || "{}");
      state.names = names && typeof names === "object" && !Array.isArray(names) ? names : {};
    }
    catch { state.names = {}; }
  }
  state.session = session; $("sessionId").value = id;
  for (const utterance of session.utterances || []) showResult(utterance);
  speakerControls();
  update(); return session;
}

async function disposeAudio() {
  state.mic?.getTracks().forEach(track => track.stop());
  state.mic = null;
  state.processor?.disconnect(); state.processor = null;
  const context = state.context; state.context = null;
  if (context && context.state !== "closed") await context.close();
  for (const pending of state.flushes.values()) pending.reject(new Error("Microphone stopped."));
  state.flushes.clear(); update();
}

function closeIfIdle() {
  if (!state.mic && !state.pending.size && !state.ending && state.ws) state.ws.close();
}
function sendCommit() {
  if (!state.samples || state.ws?.readyState !== WebSocket.OPEN) return;
  const id = state.segment || crypto.randomUUID();
  state.pending.set(id, {started: performance.now()});
  state.ws.send(JSON.stringify({type: "commit", utterance_id: id}));
  state.samples = 0; state.segment = null; state.gate?.reset();
  $("draft").hidden = true; update();
}
function discardSegment() {
  if (state.samples && state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({type: "discard"}));
  state.samples = 0; state.segment = null; state.gate?.reset(); $("draft").hidden = true;
}
function captureFrame(buffer) {
  if (!state.segment) {
    state.segment = crypto.randomUUID(); state.previewAt = performance.now();
    if (state.delivery !== "manual") showDraft("Listening…", "Waiting for sentence pause; speaker match pending");
  }
  state.ws.send(buffer); state.samples += buffer.byteLength / 2;
}
function flush(pause = false) {
  if (!state.processor) return Promise.resolve();
  const id = crypto.randomUUID();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { state.flushes.delete(id); reject(new Error("Microphone flush timed out; stop and reconnect.")); }, 5000);
    state.flushes.set(id, {resolve: () => { clearTimeout(timer); resolve(); }, reject: error => { clearTimeout(timer); reject(error); }});
    state.processor.port.postMessage({type: "flush", id, pause});
  });
}
async function stopMic() {
  await flush(true);
  if (state.delivery !== "manual" && state.gate.voiced < 4000) discardSegment(); else sendCommit();
  await disposeAudio(); closeIfIdle();
  notice("Microphone stopped. Waiting for any submitted sentences to finish.");
}

function connect(options) {
  const url = new URL(sessionPath("/stream"), location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(url); state.ws = ws;
  return new Promise((resolve, reject) => {
    let started = false;
    const timer = setTimeout(() => { reject(new Error("Stream connection timed out.")); ws.close(); }, 15000);
    ws.onmessage = ({data}) => {
      const message = JSON.parse(data);
      if (message.type === "ready") ws.send(JSON.stringify({type: "start", format: "pcm_s16le", sample_rate: 16000, channels: 1, ...options}));
      else if (message.type === "started") { started = true; clearTimeout(timer); resolve(); }
      else if (message.type === "partial" && message.client_utterance_id === state.segment) {
        if (message.text) showDraft(message.text);
      } else if (message.type === "partial_error" && message.client_utterance_id === state.segment) {
        notice(`Live preview failed; final transcription will still be attempted: ${message.error}`, true);
      }
      else if (message.type === "utterance_queued") {
        const pending = state.pending.get(message.client_utterance_id);
        if (pending) pending.job_id = message.job_id;
        notice(`Processing sentence · job ${message.job_id}`);
      } else if (message.type === "utterance" || message.type === "utterance_error") {
        const pending = state.pending.get(message.client_utterance_id);
        state.pending.delete(message.client_utterance_id);
        if (message.type === "utterance") {
          showResult(message.result, pending ? performance.now() - pending.started : null);
          notice(`Sentence transcribed · ${speakerLabel(message.result.speaker_id)}.`);
        } else notice(message.error, true);
        update(); closeIfIdle();
      } else if (message.type === "session_ended") {
        state.session = message.session; state.ending = false;
        notice("Session ended."); ws.close(); update();
      } else if (message.type === "error") {
        notice(message.error, true);
        if (!started) { clearTimeout(timer); reject(new Error(message.error)); ws.close(); }
      }
    };
    ws.onerror = () => { clearTimeout(timer); reject(new Error("WebSocket connection failed.")); };
    ws.onclose = () => {
      clearTimeout(timer);
      if (!started) reject(new Error("Stream closed before starting."));
      if (state.ws !== ws) return;
      state.ws = null;
      if (state.mic) void disposeAudio().catch(error => notice(error.message, true));
      if (state.pending.size) notice("Connection closed with pending sentences. Refresh the transcript to recover completed results.", true);
      state.pending.clear(); state.samples = 0; state.ending = false; update();
      state.segment = null; $("draft").hidden = true;
    };
  });
}

async function startMic() {
  const options = settings();
  state.delivery = $("delivery").value;
  state.gate = new SentenceGate({threshold: Number($("sensitivity").value), pause: Number($("pause").value)});
  options.partials = state.delivery === "live";
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error("Microphone access requires HTTPS or localhost.");
  try {
    state.mic = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}, video: false});
    try { state.context = new AudioContext({sampleRate: 16000}); }
    catch { state.context = new AudioContext(); }
    await state.context.resume();
    await state.context.audioWorklet.addModule("/v2/test/pcm-worklet.js");
    await connect(options);
    state.processor = new AudioWorkletNode(state.context, "pcm16-capture");
    state.processor.port.onmessage = ({data}) => {
      if (data.type === "flushed") { state.flushes.get(data.id)?.resolve(); state.flushes.delete(data.id); return; }
      if (data.type !== "audio" || state.ws?.readyState !== WebSocket.OPEN) return;
      if (state.pending.size >= 4) {
        discardSegment();
        notice("Transcription cannot keep up. Microphone stopped; submitted sentences will finish. Restart after they drain.", true);
        void disposeAudio().then(closeIfIdle); return;
      }
      if (state.ws.bufferedAmount > 2 * 1024 * 1024) {
        notice("Connection is too slow. Microphone stopped; the current unsent sentence was discarded.", true);
        state.samples = 0; state.ws.close(); void disposeAudio(); return;
      }
      if (state.delivery === "manual") {
        captureFrame(data.buffer);
        if (state.samples >= 55 * 16000) sendCommit();
      } else {
        const {frames, boundary} = state.gate.push(data.buffer);
        for (const frame of frames) captureFrame(frame);
        if (boundary === "commit") sendCommit();
        else if (boundary === "discard") discardSegment();
        else if (state.delivery === "live" && state.samples >= 8000 && performance.now() - state.previewAt >= 1500 && !state.pending.size) {
          state.ws.send(JSON.stringify({type: "preview", utterance_id: state.segment}));
          state.previewAt = performance.now();
        }
      }
      update();
    };
    const source = state.context.createMediaStreamSource(state.mic);
    const silent = state.context.createGain(); silent.gain.value = 0;
    source.connect(state.processor); state.processor.connect(silent); silent.connect(state.context.destination);
    notice(`Microphone on. ${state.delivery === "manual" ? "Speak, then click Send sentence." : "Speak naturally; pauses submit sentences automatically."}`);
  } catch (error) {
    await disposeAudio(); state.ws?.close(); throw error;
  }
}

$("create").onclick = () => run(async () => {
  const session = await api("/v2/sessions", jsonPost({capture_voiceprints: $("capture").checked, metadata: {client: "browser-v2-tester"}}));
  await loadSession(session.session_id, true); notice("Session created.");
});
$("attach").onclick = () => run(async () => {
  const id = $("sessionId").value.trim(); if (!id) throw new Error("Enter a session ID.");
  await loadSession(id, true); notice("Session opened.");
});
$("mode").onchange = () => { if ($("mode").value !== "parakeet" && !$("language").value) $("language").value = "en"; update(); };
$("file").onchange = update;
$("speakerSelect").onchange = () => { $("speakerName").value = state.names[$("speakerSelect").value] || ""; update(); };
$("nameSpeaker").onclick = () => {
  const id = $("speakerSelect").value; if (!id) return;
  const name = $("speakerName").value.trim();
  if (name) state.names[id] = name; else delete state.names[id];
  try { localStorage.setItem(`parascribe-names-${state.session.session_id}`, JSON.stringify(state.names)); }
  catch { notice("Name applied, but browser storage is unavailable.", true); }
  for (const result of [...state.results.values()]) showResult(result);
};
for (const tab of ["live", "final"]) $(`${tab}Tab`).onclick = () => {
  for (const name of ["live", "final"]) {
    $(`${name}Tab`).setAttribute("aria-selected", String(name === tab));
    $(`${name}Panel`).hidden = name !== tab;
  }
};
$("start").onclick = () => run(startMic);
$("commit").onclick = () => run(async () => { await flush(); sendCommit(); });
$("stop").onclick = () => run(stopMic);
$("refresh").onclick = () => run(async () => { await loadSession(state.session.session_id); notice("Transcript refreshed."); });
$("delete").onclick = () => run(async () => {
  const result = await api(sessionPath("/voiceprints"), {method: "DELETE"});
  state.session.capture_voiceprints = false;
  notice(`Deleted ${result.deleted_voiceprints} voice signatures. Transcript retained.`);
});
$("end").onclick = () => run(async () => {
  const deleteVoiceprints = $("deleteOnEnd").checked;
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ending = true;
    try {
      if (state.mic) {
        await flush(true);
        if (state.delivery !== "manual" && state.gate.voiced < 4000) discardSegment(); else sendCommit();
        await disposeAudio();
      }
      state.ws.send(JSON.stringify({type: "end", delete_voiceprints: deleteVoiceprints}));
      notice("Finishing pending sentences and ending the session…");
    } catch (error) { state.ending = false; throw error; }
  } else {
    state.session = await api(sessionPath("/end"), jsonPost({delete_voiceprints: deleteVoiceprints}));
    notice("Session ended.");
  }
});
$("upload").onclick = () => run(async () => {
  const options = settings(), file = $("file").files[0];
  if (!file) throw new Error("Select an audio file.");
  const form = new FormData(); form.append("file", file); form.append("asr_mode", options.asr_mode);
  if (options.language) form.append("language", options.language);
  const started = performance.now();
  const queued = await api(sessionPath("/sentences"), {method: "POST", body: form});
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    const job = await api(`/v1/jobs/${encodeURIComponent(queued.job_id)}`);
    if (job.status === "done") { showResult(job.result, performance.now() - started); notice("Uploaded sentence transcribed."); return; }
    if (job.status === "error") throw new Error(job.error || "Transcription failed.");
    notice(`Job ${queued.job_id}: ${job.stage || job.status}`);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error(`Still processing job ${queued.job_id}. Refresh the transcript later.`);
});
$("download").onclick = () => run(async () => {
  await loadSession(state.session.session_id);
  const blob = new Blob([JSON.stringify({session: state.session, speaker_names: state.names, results: [...state.results.values()]}, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = `parascribe-${state.session.session_id}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("pagehide", () => { state.mic?.getTracks().forEach(track => track.stop()); state.ws?.close(); });

update();
try {
  const health = await api("/health"); state.cohere = !!health.cohere_configured;
  notice(`Server ready. Cohere ${state.cohere ? "configured" : "not configured; Parakeet available"}.${window.isSecureContext ? "" : " Use HTTPS or localhost for microphone access; uploads still work."}`);
} catch (error) { notice(`Cannot reach the server: ${error.message}`, true); }
