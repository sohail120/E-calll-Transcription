import soundcard as sc
import soundfile as sf
import speech_recognition as sr
import time
import whisper

r = sr.Recognizer()

OUTPUT_FILE_NAME = "out.wav"    # file name.
SAMPLE_RATE = 48000              # [Hz]. sampling rate.
RECORD_SEC = 5                  # [sec]. duration recording audio.
model = whisper.load_model("base")

while True :
   with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
       data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
       sf.write(file=OUTPUT_FILE_NAME, data=data[:, 0], samplerate=SAMPLE_RATE)
       time.sleep(1)
       with sr.AudioFile(OUTPUT_FILE_NAME) as source: 
           audio_text = r.listen(source)
           try:
              result = model.transcribe(source,fp16=False)
              print(result["text"])
           except:
               print('Sorry.. run again...')