import threading
import time
import soundcard as sc
import soundfile as sf
import whisper
import speech_recognition as sr
import sounddevice as sd
import wavio as wv
import whisper
import multiprocessing
import os
# import redis

SPEAKER_OUTPUT = "speakerChunks.wav"    # file name.
MIC_OUTPUT = "speakerChunks.wav"    # file name.

SAMPLE_RATE = 48000              # [Hz]. sampling rate.
RECORD_SEC = 5                  # [sec]. duration recording audio.
model = whisper.load_model("base.en")
recognizer = sr.Recognizer()
SAMPLE_RATE = 48000   


def speaker_audio_recording_service():
      while True:
          with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
             data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
             sf.write(file=SPEAKER_OUTPUT, data=data[:, 0], samplerate=SAMPLE_RATE)

def speaker_audio_transcription_Service():
      while True:
          audio = whisper.load_audio(SPEAKER_OUTPUT)
          audio = whisper.pad_or_trim(audio)
          mel = whisper.log_mel_spectrogram(audio).to(model.device)
          _, probs = model.detect_language(mel)
          options = whisper.DecodingOptions(fp16 = False)
          result = whisper.decode(model, mel, options)
          print("Incoming Call Person",result.text)
    


# Create two threads
thread1 = threading.Thread(target=speaker_audio_recording_service)
thread2 = threading.Thread(target=speaker_audio_transcription_Service)

# Start the threads
thread1.start()
thread2.start()


