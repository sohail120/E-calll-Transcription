import whisper

model = whisper.load_model("base")
audio = whisper.load_audio("out.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
_, probs = model.detect_language(mel)
options = whisper.DecodingOptions(fp16 = False)
result = whisper.decode(model, mel, options)
print(result.text)