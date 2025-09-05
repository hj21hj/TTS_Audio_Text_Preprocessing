// app.js
// Robust ESM -> UMD fallback loader for Transformers.js (Whisper)
const $ = (sel) => document.querySelector(sel);
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

function logStatus(msg) {
  statusEl.textContent = `상태: ${msg}`;
  console.log(msg);
}

async function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

async function loadTransformers() {
  try {
    // Try ESM first
    logStatus("Transformers.js ESM 로딩 중...");
    const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1');
    logStatus("ESM 로딩 성공");
    return { pipeline: mod.pipeline, env: mod.env };
  } catch (e) {
    console.warn("CDN ESM 로딩 실패 — UMD 폴백 시도", e);
    logStatus("ESM 실패 → UMD 폴백 로딩 중...");
    await loadScript('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1/dist/transformers.min.js');
    const { pipeline, env } = window.transformers;
    logStatus("UMD 로딩 성공");
    return { pipeline, env };
  }
}

function secToSrtTime(s) {
  const sign = s < 0 ? "-" : "";
  s = Math.max(0, s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s - Math.floor(s)) * 1000);
  return `${sign}${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')},${String(ms).padStart(3,'0')}`;
}

function secToVttTime(s) {
  const sign = s < 0 ? "-" : "";
  s = Math.max(0, s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s - Math.floor(s)) * 1000);
  return `${sign}${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
}

function buildSRT(segments) {
  let idx = 1;
  return segments.map(seg => {
    const [start, end] = seg.timestamp || seg.time || [0, 0];
    const text = (seg.text || "").trim();
    const block = `${idx}\n${secToSrtTime(start)} --> ${secToSrtTime(end)}\n${text}\n`;
    idx++;
    return block;
  }).join("\n");
}

function buildVTT(segments) {
  let out = "WEBVTT\n\n";
  let idx = 1;
  out += segments.map(seg => {
    const [start, end] = seg.timestamp || seg.time || [0, 0];
    const text = (seg.text || "").trim();
    const block = `${secToVttTime(start)} --> ${secToVttTime(end)}\n${text}\n`;
    idx++;
    return block;
  }).join("\n");
  return out;
}

function downloadFile(filename, content) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function enableDownloads(enabled) {
  [btnTxt, btnSrt, btnVtt, btnJson].forEach(b => b.disabled = !enabled);
}

function resetAll() {
  gFile = null;
  gSegments = null;
  txtOut.value = "";
  audioEl.src = "";
  btnTranscribe.disabled = true;
  enableDownloads(false);
  logStatus("대기 중");
  fileInput.value = "";
}

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFile(e.dataTransfer.files[0]);
  }
});
fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  gFile = file;
  const url = URL.createObjectURL(file);
  audioEl.src = url;
  btnTranscribe.disabled = !gASR;
  logStatus(`파일 준비됨: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`);
}

let transformers = null;

btnInit.addEventListener("click", async () => {
  btnInit.disabled = true;
  try {
    transformers = await loadTransformers();
    const { pipeline, env } = transformers;

    // Optional engine tweaks
    env.allowRemoteModels = true;
    // env.backends.onnx.wasm.numThreads = 1; // if needed for stability on low-end devices

    const modelId = modelSelect.value;
    const quantized = quantizedInput.checked;

    logStatus(`모델 로딩 중: ${modelId} ${quantized ? "(quantized)" : ""} ...`);

    gASR = await pipeline('automatic-speech-recognition', modelId, {
      quantized,
      progress_callback: (data) => {
        // Possible fields: status, name, file, loaded, total, progress
        const p = (data.progress != null) ? Math.round(data.progress * 100) + "%" : "";
        logStatus(`모델 받는 중: ${data.status || ""} ${p}`.trim());
      }
    });

    logStatus(`모델 준비 완료: ${modelId}`);
    if (gFile) btnTranscribe.disabled = false;
  } catch (err) {
    console.error(err);
    logStatus("모델 로딩 실패. 네트워크 또는 메모리 상태를 확인하세요.");
    btnInit.disabled = false;
  }
});

btnTranscribe.addEventListener("click", async () => {
  if (!gASR || !gFile) return;
  btnTranscribe.disabled = true;
  enableDownloads(false);
  txtOut.value = "";
  gSegments = null;

  const lang = langSelect.value;
  const chunkLen = Math.max(10, Math.min(120, parseInt(chunkLengthInput.value || "30", 10)));

  logStatus("음성 처리 시작...");
  try {
    const out = await gASR(gFile, {
      return_timestamps: true,
      chunk_length_s: chunkLen,
      stride_length_s: 5,
      language: lang === "auto" ? undefined : lang,
      // callback for chunk progress (if exposed)
      callback_function: (data) => {
        if (data?.chunks?.length) {
          logStatus(`진행 중... 세그먼트 ${data.chunks.length}개 생성됨`);
        }
      }
    });

    const text = (out?.text || "").trim();
    const segments = out?.chunks || [];
    txtOut.value = text;
    gSegments = segments;

    logStatus(`완료: 총 세그먼트 ${segments.length}개`);
    enableDownloads(true);
  } catch (err) {
    console.error(err);
    logStatus("실패: 처리 중 오류가 발생했습니다. (긴 파일/메모리 부족일 수 있음)");
  } finally {
    btnTranscribe.disabled = false;
  }
});

btnClear.addEventListener("click", resetAll);

btnTxt.addEventListener("click", () => {
  const base = (gFile?.name || "transcript").replace(/\.[^/.]+$/, "");
  downloadFile(`${base}.txt`, txtOut.value || "");
});
btnSrt.addEventListener("click", () => {
  const base = (gFile?.name || "transcript").replace(/\.[^/.]+$/, "");
  const srt = buildSRT(gSegments || []);
  downloadFile(`${base}.srt`, srt);
});
btnVtt.addEventListener("click", () => {
  const base = (gFile?.name || "transcript").replace(/\.[^/.]+$/, "");
  const vtt = buildVTT(gSegments || []);
  downloadFile(`${base}.vtt`, vtt);
});
btnJson.addEventListener("click", () => {
  const base = (gFile?.name || "transcript").replace(/\.[^/.]+$/, "");
  const json = JSON.stringify(gSegments || [], null, 2);
  downloadFile(`${base}.segments.json`, json);
});

// Initial status
resetAll();
