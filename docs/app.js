// app.js (Fix1): support File/Blob by converting to Float32 PCM, set wasmPaths, better errors.
const $ = (s) => document.querySelector(s);
const statusEl = $("#status");
const audioEl = $("#audio");
const fileInput = $("#file-input");
const dropzone = $("#dropzone");
const modelSelect = $("#model-select");
const langSelect = $("#lang-select");
const chunkLengthInput = $("#chunk-length");
const quantizedInput = $("#quantized");
const btnInit = $("#btn-init");
const btnTranscribe = $("#btn-transcribe");
const btnClear = $("#btn-clear");
const txtOut = $("#result-text");
const btnTxt = $("#btn-download-txt");
const btnSrt = $("#btn-download-srt");
const btnVtt = $("#btn-download-vtt");
const btnJson = $("#btn-download-json");

let gFile = null;
let gASR = null;
let gSegments = null;
let transformers = null;

function logStatus(msg) {
  statusEl.textContent = `상태: ${msg}`;
  console.log(msg);
}

async function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Script load failed: ${src}`));
    document.head.appendChild(s);
  });
}

async function loadTransformers() {
  try {
    logStatus("Transformers.js ESM 로딩 중...");
    const mod = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1");
    logStatus("ESM 로딩 성공");
    return { pipeline: mod.pipeline, env: mod.env };
  } catch (e) {
    console.warn("CDN ESM 로딩 실패 — UMD 폴백 시도", e);
    logStatus("ESM 실패 → UMD 폴백 로딩 중...");
    await loadScript("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1/dist/transformers.min.js");
    if (!window.transformers) throw new Error("UMD 로딩 실패");
    logStatus("UMD 로딩 성공");
    return { pipeline: window.transformers.pipeline, env: window.transformers.env };
  }
}

// Convert uploaded File/Blob to Float32 PCM (mono).
async function fileToPCM(file) {
  const arrayBuffer = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ac = new AC({ sampleRate: 16000 }); // resample target
  if (ac.state === "suspended") await ac.resume();
  const audioBuffer = await ac.decodeAudioData(arrayBuffer.slice(0));
  // Mixdown to mono
  const ch0 = audioBuffer.getChannelData(0);
  let pcm = ch0;
  if (audioBuffer.numberOfChannels > 1) {
    const ch1 = audioBuffer.getChannelData(1);
    const len = Math.min(ch0.length, ch1.length);
    const mixed = new Float32Array(len);
    for (let i = 0; i < len; i++) mixed[i] = 0.5 * (ch0[i] + ch1[i]);
    pcm = mixed;
  }
  try { await ac.close(); } catch {}
  return pcm;
}

function secToSrtTime(s) {
  s = Math.max(0, s || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s - Math.floor(s)) * 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}
function secToVttTime(s) {
  s = Math.max(0, s || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s - Math.floor(s)) * 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}
function buildSRT(segments) {
  return (segments || []).map((seg, i) => {
    const [start, end] = seg.timestamp || seg.time || [0, 0];
    const text = (seg.text || "").trim();
    return `${i + 1}\n${secToSrtTime(start)} --> ${secToSrtTime(end)}\n${text}\n`;
  }).join("\n");
}
function buildVTT(segments) {
  let out = "WEBVTT\n\n";
  out += (segments || []).map(seg => {
    const [start, end] = seg.timestamp || seg.time || [0, 0];
    const text = (seg.text || "").trim();
    return `${secToVttTime(start)} --> ${secToVttTime(end)}\n${text}\n`;
  }).join("\n");
  return out;
}
function downloadFile(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function enableDownloads(enabled) {
  [btnTxt, btnSrt, btnVtt, btnJson].forEach(b => b.disabled = !enabled);
}
function resetAll() {
  gFile = null; gSegments = null; txtOut.value = ""; audioEl.src = "";
  btnTranscribe.disabled = true; enableDownloads(false); logStatus("대기 중"); fileInput.value = "";
}

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", (e) => { const f = e.target.files?.[0]; if (f) handleFile(f); });

function handleFile(file) {
  gFile = file;
  const url = URL.createObjectURL(file);
  audioEl.src = url;
  btnTranscribe.disabled = !gASR;
  logStatus(`파일 준비됨: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`);
}

btnInit.addEventListener("click", async () => {
  btnInit.disabled = true;
  try {
    transformers = await loadTransformers();
    const { pipeline, env } = transformers;

    // Ensure WASM assets load from CDN path
    env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1/dist/";
    env.allowRemoteModels = true;

    const modelId = modelSelect.value;
    const quantized = quantizedInput.checked;

    logStatus(`모델 로딩 중: ${modelId} ${quantized ? "(quantized)" : ""} ...`);
    gASR = await pipeline("automatic-speech-recognition", modelId, {
      quantized,            // v2 option (ignored if unsupported)
      dtype: quantized ? "q8" : undefined, // hint for some builds
      progress_callback: (data) => {
        const p = (data.progress != null) ? ` ${Math.round(data.progress*100)}%` : "";
        logStatus(`모델 받는 중: ${data.status || ""}${p}`.trim());
      },
    });

    logStatus(`모델 준비 완료: ${modelId}`);
    if (gFile) btnTranscribe.disabled = false;
  } catch (err) {
    console.error(err);
    logStatus(`모델 로딩 실패: ${err.message}`);
    btnInit.disabled = false;
  }
});

btnTranscribe.addEventListener("click", async () => {
  if (!gASR || !gFile) { return; }
  btnTranscribe.disabled = true; enableDownloads(false); txtOut.value = ""; gSegments = null;
  const lang = langSelect.value;
  const chunkLen = Math.max(10, Math.min(120, parseInt(chunkLengthInput.value || "30", 10)));
  logStatus("음성 처리 시작...");
  try {
    const pcm = await fileToPCM(gFile);
    const out = await gASR(pcm, {
      return_timestamps: true,
      chunk_length_s: chunkLen,
      stride_length_s: 5,
      language: lang === "auto" ? undefined : lang,
      task: "transcribe",
    });
    const text = (out?.text || "").trim();
    const segments = out?.chunks || [];
    txtOut.value = text; gSegments = segments;
    logStatus(`완료: 총 세그먼트 ${segments.length}개`);
    enableDownloads(true);
  } catch (err) {
    console.error(err);
    logStatus(`실패: ${err.message || "처리 중 오류"}`);
  } finally {
    btnTranscribe.disabled = false;
  }
});

btnClear.addEventListener("click", resetAll);
btnTxt.addEventListener("click", () => { const base = (gFile?.name || "transcript").replace(/\.[^/.]+$/, ""); downloadFile(`${base}.txt`, txtOut.value || ""); });
btnSrt.addEventListener("click", () => { const base = (gFile?.name || "transcript").replace(/\.[^/.]+$/, ""); const srt = buildSRT(gSegments || []); downloadFile(`${base}.srt`, srt); });
btnVtt.addEventListener("click", () => { const base = (gFile?.name || "transcript").replace(/\.[^/.]+$/, ""); const vtt = buildVTT(gSegments || []); downloadFile(`${base}.vtt`, vtt); });
btnJson.addEventListener("click", () => { const base = (gFile?.name || "transcript").replace(/\.[^/.]+$/, ""); const json = JSON.stringify(gSegments || [], null, 2); downloadFile(`${base}.segments.json`, json); });

resetAll();
