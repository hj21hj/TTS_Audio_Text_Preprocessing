# Audio_Text_Preprocessing
한국어 TTS를 위한 오디오-텍스트 데이터 전처리 코드를 간략하게 정리했습니다.
캐글의 한국어 단일 화자 음성 데이터셋 KSS dataset을 사용해 한국어 TTS를 구현했습니다. 

`preprocessing_EDA.ipynb`는 KSS dataset을 이해하기 위한 시각화와 간단한 text cleaning 예제를 담고 있습니다.
`audio_text_preprocessing.ipynb`는 한국어 TTS를 수행하기 위한 오디오파일 및 텍스트 파일의 필수 전처리 코드를 담고 있습니다.
`main.py`는 Tacotron2와 WaveGlow를 각각 학습하고 음성을 합성할 수 있는 총괄 실행 코드입니다.

음향모델 및 보코더 구현체는 아래 레포지토리를 사용했습니다. 

- [Tacotron2] https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS/tree/master
- [WaveGlow] https://github.com/NVIDIA/waveglow
