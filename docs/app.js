// app.js (Fix2)
const $ = (s) => document.querySelector(s);
const statusEl = $("#status");
const debugWrap = $("#debug");
const debugPre = $("#debug-pre");
const showDebug = $("#show-debug");
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
  if (showDebug.checked) {
    debugPre.textContent += `[STATUS] ${msg}\n`;
  }
}
function logDebug(obj, label="DEBUG") {
  if (!showDebug.checked) return;
  try {
    debugPre.textContent += `[${label}] ${JSON.stringify(obj, null, 2)}\n`;
  } catch {
    debugPre.textContent += `[${label}] ${String(obj)}\n`;
  }
}
function clearDebug() {
  debugPre.textContent = "";
}

async function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src; s.async = true;
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

// Helpers: normalize & trim silence
function normalizePeakPCM(pcm, target=0.97) {
  let peak = 0;
  for (let i=0;i<pcm.length;i++) { const v = Math.abs(pcm[i]); if (v>peak) peak = v; }
  if (peak === 0) return pcm;
  const g = target / peak;
  const out = new Float32Array(pcm.length);
  for (let i=0;i<pcm.length;i++) out[i] = Math.max(-1, Math.min(1, pcm[i]*g));
  return out;
}
function trimSilence(pcm, threshold=0.005, pad=16000) {
  // threshold ~-50dB; pad=1s at 16k
  let start=0, end=pcm.length-1;
  while (start<pcm.length && Math.abs(pcm[start]) < threshold) start++;
  while (end>start && Math.abs(pcm[end]) < threshold) end--;
  start = Math.max(0, start - pad);
  end = Math.min(pcm.length-1, end + pad);
  return pcm.slice(start, end+1);
}

// Convert uploaded File/Blob to Float32 PCM (mono, 16k)
async function fileToPCM16000(file) {
  const arrayBuffer = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  // Some browsers ignore sampleRate on regular AudioContext; use OfflineAudioContext to resample reliably.
  const tmpAC = new AC();
  const decoded = await tmpAC.decodeAudioData(arrayBuffer.slice(0));
  try { await tmpAC.close(); } catch{}
  // Mixdown to mono
  const ch0 = decoded.getChannelData(0);
  let mono = ch0;
  if (decoded.numberOfChannels > 1) {
    const ch1 = decoded.getChannelData(1);
    const len = Math.min(ch0.length, ch1.length);
    const mixed = new Float32Array(len);
    for (let i=0;i<len;i++) mixed[i] = 0.5*(ch0[i] + ch1[i]);
    mono = mixed;
  }
  // Resample to 16k
  const srcRate = decoded.sampleRate;
  if (srcRate !== 16000) {
    const duration = mono.length / srcRate;
    const oac = new OfflineAudioContext(1, Math.ceil(16000 * duration), 16000);
    const buffer = oac.createBuffer(1, mono.length, srcRate);
    buffer.copyToChannel(mono, 0);
    const src = oac.createBufferSource();
    src.buffer = buffer;
    src.connect(oac.destination);
    src.start();
    const rendered = await oac.startRendering();
    mono = rendered.getChannelData(0).slice(0);
  }
  // DC offset remove (simple)
  let mean = 0;
  for (let i=0;i<mono.length;i++) mean += mono[i];
  mean /= mono.length;
  for (let i=0;i<mono.length;i++) mono[i] -= mean;
  // normalize & trim
  mono = normalizePeakPCM(mono, 0.97);
  mono = trimSilence(mono, 0.0045, 8000);
  return mono;
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
  btnTranscribe.disabled = true; enableDownloads(false); logStatus("대기 중");
  fileInput.value = ""; clearDebug();
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
    env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1/dist/";
    env.allowRemoteModels = true;

    const modelId = modelSelect.value;
    const quantized = quantizedInput.checked;

    logStatus(`모델 로딩 중: ${modelId} ${quantized ? "(quantized)" : ""} ...`);
    gASR = await pipeline("automatic-speech-recognition", modelId, {
      quantized,
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
    const pcm = await fileToPCM16000(gFile);
    logDebug({len: pcm.length, head: Array.from(pcm.slice(0,10))}, "PCM");
    // Prefer explicit object with sampling rate
    let out = await gASR({ array: pcm, sampling_rate: 16000 }, {
      return_timestamps: true,
      chunk_length_s: chunkLen,
      stride_length_s: 5,
      language: lang === "auto" ? undefined : lang,
      task: "transcribe",
    });
    logDebug(out, "RAW_OUT");

    let text = (out?.text || "").trim();
    let segments = out?.chunks || [];

    // If auto language produced empty, retry with Korean
    if (!text && lang === "auto") {
      logStatus("자동 인식 결과가 비어, 한국어(ko)로 재시도...");
      out = await gASR({ array: pcm, sampling_rate: 16000 }, {
        return_timestamps: true,
        chunk_length_s: chunkLen,
        stride_length_s: 5,
        language: "ko",
        task: "transcribe",
      });
      text = (out?.text || "").trim();
      segments = out?.chunks || [];
      logDebug(out, "RAW_OUT_KO_RETRY");
    }

    txtOut.value = text;
    gSegments = segments;
    if (text) {
      logStatus(`완료: 총 세그먼트 ${segments.length}개`);
      enableDownloads(true);
    } else {
      logStatus("결과가 비었습니다. 파일에 유효한 발화가 없는지/언어 설정을 확인하세요.");
    }
  } catch (err) {
    console.error(err);
    logStatus(`실패: ${err.message || "처리 중 오류"}`);
    logDebug(err, "ERROR");
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
