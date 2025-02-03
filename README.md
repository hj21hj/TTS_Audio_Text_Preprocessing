<details>
<summary>🇺🇸 Click to see English README</summary>

# 🎤 TTS_Audio_Text_Preprocessing

This project provides audio-text preprocessing and model training code for Korean TTS (Text-to-Speech). 
We use the **KSS dataset**, a Korean single-speaker voice dataset from Kaggle, to implement a Tacotron2 and WaveGlow-based Korean TTS system. 

## Features
- **Audio & Text Preprocessing**: Cleans and transforms the KSS dataset for training.
- **Exploratory Data Analysis (EDA)**: Visualizes and cleans text data.
- **TTS Model Training**: Trains Tacotron2 and WaveGlow models separately.
- **Speech Synthesis**: Generates speech from input Korean text using trained models.

## Technologies Used
- **Programming Language**: Python 
- **Frameworks & Libraries**:
  - PyTorch 
  - NumPy, Pandas 
  - Matplotlib, Seaborn (for visualization)
  - Soundfile (for audio storage and conversion)
  - Tacotron2 & WaveGlow (TTS models)

## 🚀 Installation & Usage 

### 1. Setup Environment
#### (1) Install Required Libraries
Run the following command to install necessary packages:
```bash
pip install torch numpy pandas matplotlib soundfile
```

#### (2) Clone Tacotron2 & WaveGlow Repositories
```bash
git clone https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS.git
cd Tacotron2-Wavenet-Korean-TTS
```
```bash
git clone https://github.com/NVIDIA/waveglow.git
cd waveglow
```

### 2. Download Dataset
Download the KSS dataset and place it in the `Dataset/` directory.
[KSS dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

### 3. Run Preprocessing & EDA
#### (1) Data Exploration & Preprocessing
```bash
jupyter notebook preprocessing_EDA.ipynb
```
#### (2) Run Preprocessing Code
```bash
jupyter notebook audio_text_preprocessing.ipynb
```

### 4. Train Models 
#### (1) Train Tacotron2
```bash
python main.py --mode train_tacotron
```
#### (2) Train WaveGlow
```bash
python main.py --mode train_waveglow
```

### 5. Speech Synthesis 
Generate speech from text using trained models:
```bash
python main.py --mode synthesize --text "Hello. This is a Korean TTS project."
```
</details>

---

# 🎵 TTS_Audio_Text_Preprocessing

한국어 TTS(Text-to-Speech)를 위한 오디오-텍스트 데이터 전처리 및 모델 학습 코드입니다! 
Kaggle의 한국어 단일 화자 음성 데이터셋 **KSS dataset**을 사용하여 Tacotron2와 WaveGlow 기반의 한국어 TTS를 구현합니다.

## ✨ 주요 기능 
- **오디오 및 텍스트 데이터 전처리** : KSS dataset을 정제하고, 모델 학습을 위한 적절한 형식으로 변환합니다.
- **EDA(탐색적 데이터 분석)** : 데이터셋을 시각화하고, 텍스트 정제 과정을 포함합니다.
- **TTS 모델 학습** : Tacotron2 및 WaveGlow 모델을 각각 학습시킵니다.
- **음성 합성** : 학습된 모델을 사용하여 입력된 한국어 텍스트의 음성을 생성합니다.

## 🔧 사용된 기술 
- **프로그래밍 언어**: Python 
- **프레임워크 및 라이브러리**:
  - PyTorch 
  - NumPy, Pandas 
  - Matplotlib, seaborn (시각화용)
  - Soundfile (오디오 저장 및 변환)
  - Tacotron2 및 WaveGlow (TTS 모델)

## 🚀 설치 및 실행 방법 

### 1. 환경 설정
#### (1) 필요한 라이브러리 설치
```bash
pip install torch numpy pandas matplotlib soundfile
```

#### (2) Tacotron2 및 WaveGlow 레포지토리 클론
```bash
git clone https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS.git
cd Tacotron2-Wavenet-Korean-TTS
```
```bash
git clone https://github.com/NVIDIA/waveglow.git
cd waveglow
```

### 2. 데이터셋 다운로드 📂
[KSS dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)을 다운로드한 후, `Dataset/` 디렉토리에 저장합니다.

### 3. 데이터 전처리 및 분석 실행
#### (1) 데이터 탐색 및 전처리 실행
```bash
jupyter notebook preprocessing_EDA.ipynb
```
#### (2) 데이터 전처리 코드 실행
```bash
jupyter notebook audio_text_preprocessing.ipynb
```

### 4. 모델 학습 
#### (1) Tacotron2 학습
```bash
python main.py --mode train_tacotron
```
#### (2) WaveGlow 학습
```bash
python main.py --mode train_waveglow
```

### 5. 음성 합성 
```bash
python main.py --mode synthesize --text "안녕하세요. 한국어 TTS 프로젝트입니다."
```

## 디렉토리 구조 
```
📂 Audio_Text_Preprocessing
├── preprocessing_EDA.ipynb  # 데이터 탐색 및 전처리 시각화 코드
├── audio_text_preprocessing.ipynb  # 오디오 및 텍스트 전처리 코드
├── main.py  # 모델 학습 및 음성 합성 실행 파일
├── tacotron2/
├── waveglow/
├── Dataset/  # KSS dataset 저장 디렉토리
├── res/
│   ├── output/  # 생성된 음성 파일 저장 디렉토리
│   ├── logs/  # 학습 로그 저장 디렉토리
│   ├── checkpoints/  # 모델 체크포인트 저장 디렉토리
```

## 🤝 기여 방법 
1. 해당 레포지토리를 포크합니다.
2. 새로운 브랜치를 생성합니다: `git checkout -b feature-branch`
3. 변경 사항을 커밋합니다: `git commit -m "설명 추가"`
4. 브랜치를 푸시합니다: `git push origin feature-branch`
5. Pull Request를 생성합니다.

## 📚 참고 자료 
- [Tacotron2](https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS/tree/master)
- [WaveGlow](https://github.com/NVIDIA/waveglow)

## 📜 라이선스 
본 프로젝트는 MIT 라이선스를 따릅니다.

