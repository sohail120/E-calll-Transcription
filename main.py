import multiprocessing
import soundcard as sc
import soundfile as sf
import whisper
import speech_recognition as sr

SPEAKER_OUTPUT = "speakerChunks.wav"    # File name for speaker audio.
MIC_OUTPUT = "speakerChunks.wav"         # File name for microphone audio.

SAMPLE_RATE = 48000                      # [Hz]. Sampling rate.
RECORD_SEC = 5                          # [sec]. Duration for audio recording.
model = whisper.load_model("base")
recognizer = sr.Recognizer()


def speaker_audio_recording_service():
    while True:
        with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
            data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
            sf.write(file=SPEAKER_OUTPUT, data=data[:, 0], samplerate=SAMPLE_RATE)

def speaker_audio_transcription_service():
    while True:
        audio = whisper.load_audio(SPEAKER_OUTPUT)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        print("Incoming Call Person", result.text)

if __name__ == "__main__":
    # Create separate processes for audio recording and transcription
    record_process = multiprocessing.Process(target=speaker_audio_recording_service)
    transcription_process = multiprocessing.Process(target=speaker_audio_transcription_service)

    # Start the processes
    record_process.start()
    transcription_process.start()

    # Wait for the processes to finish (this code will run indefinitely until manually terminated)
    record_process.join()
    transcription_process.join()
