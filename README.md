# 🔉Audio-Emotion-Detection

This project aims to detect emotions from audio files using machine learning models. The primary model used is the Wav2Vec2 model from HuggingFace's Transformers library.


## Requirements

- Python 3.11.7
- torch
- torchaudio
- transformers
- accelerate
- pandas
- librosa
- matplotlib

## Setup

1. Clone the repository.
2. Install the required packages:
    ```sh
    pip install torch torchaudio transformers accelerate pandas librosa matplotlib
    ```

## Usage

Open the `EmoRec.ipynb` notebook and run the cells to analyze emotions in audio files. The notebook includes functions to:

- Extract audio features and generate spectrograms.
- Setup and use the HuggingFace Wav2Vec2 model for emotion recognition.
- Analyze audio files and print detected emotions and confidence scores.

## Example

```python
audio_file = "Crema_data\\1091_WSI_HAP_XX.wav"

results = analyze_audio_emotion(
    audio_file, 
    use_huggingface=True,
    use_resnet=False,
    save_spectrogram=True
)

# Print results
print(f"Audio file: {audio_file}")
if 'huggingface_emotion' in results:
    print(f"Detected emotion (HuggingFace): {results['huggingface_emotion']}")
    print("Emotion confidence scores:")
    for emotion, score in results['huggingface_scores'].items():
        print(f"  {emotion}: {score:.4f}")

if 'spectrogram_path' in results:
    print(f"Spectrogram saved to: {results['spectrogram_path']}")
