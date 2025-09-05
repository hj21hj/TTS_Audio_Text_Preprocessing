# Fix1: 버튼 눌러도 동작하지 않을 때 대처
- **원인**: Whisper 파이프라인이 브라우저에서 `File/Blob` 입력을 직접 받지 못하는 경우가 있어, 내부적으로 무시되거나 에러 없이 멈춥니다.
- **해결**: 업로드한 파일을 **Float32 PCM**으로 변환해 전달하도록 수정. 또한 **WASM 경로(wasmPaths)** 를 CDN으로 명시해, GitHub Pages 등에서 로딩 실패를 방지했습니다.

## 배포
이 폴더를 GitHub Pages에 그대로 올리면 됩니다. (Settings → Pages)
