import time
import soundcard as sc
import soundfile as sf
import whisper

OUTPUT_FILE_NAME = "out.wav"    # file name.
SAMPLE_RATE = 48000              # [Hz]. sampling rate.
RECORD_SEC = 3                  # [sec]. duration recording audio.
model = whisper.load_model("base")

while True:
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
       data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
       sf.write(file=OUTPUT_FILE_NAME, data=data[:, 0], samplerate=SAMPLE_RATE)
       time.sleep(1)
       audio = whisper.load_audio("out.wav")
       audio = whisper.pad_or_trim(audio)
       mel = whisper.log_mel_spectrogram(audio).to(model.device)
       _, probs = model.detect_language(mel)
       options = whisper.DecodingOptions(fp16 = False)
       result = whisper.decode(model, mel, options)
       print(result.text)

