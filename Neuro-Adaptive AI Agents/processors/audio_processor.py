# processors/audio_processor.py

import os
import librosa # type: ignore
from pydub import AudioSegment # type: ignore
from models.speech_to_text import transcribe_audio
from models.emotion_recognition import detect_text_emotion
import webrtcvad # type: ignore
import wave

TEMP_AUDIO_PATH = "data/converted_audio.wav"

def convert_audio_to_vosk_format(input_path, output_path=TEMP_AUDIO_PATH):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def is_speech(audio_path):
    vad = webrtcvad.Vad(2)
    with wave.open(audio_path, 'rb') as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio must be WAV format: mono, 16-bit, 16kHz")
        frames = wf.readframes(wf.getnframes())
        frame_duration = 30
        frame_size = int(16000 * frame_duration / 1000) * 2
        num_frames = len(frames) // frame_size
        speech_frames = sum(
            1 for i in range(num_frames)
            if vad.is_speech(frames[i * frame_size:(i + 1) * frame_size], 16000)
        )
        speech_ratio = speech_frames / max(1, num_frames)
        return speech_ratio > 0.3

def process_audio(audio_path):
    result = {}
    formatted_path = convert_audio_to_vosk_format(audio_path)
    result['is_speech'] = is_speech(formatted_path)

    if result['is_speech']:
        transcribed = transcribe_audio(formatted_path)
        emotion, score, importance = detect_text_emotion(transcribed)  # ✅ Unpack all three

        result.update({
            'transcribed': transcribed,
            'emotion': emotion,
            'emotion_score': score,
            'audio_text': f"Emotion: {emotion}, Text: {transcribed}",
            'importance': round(importance, 3)  # ✅ Use importance from emotion detection
        })

    else:
        result.update({
            'transcribed': "",
            'emotion': "none",
            'emotion_score': 0,
            'audio_text': "Non-speech sound",
            'importance': 0.7  # Might be useful ambient audio (e.g., siren)
        })

    return result

