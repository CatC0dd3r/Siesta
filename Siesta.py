# Siesta V0.0.2
import sys
import os
import torch
import datetime
import sys
import webbrowser
import subprocess
import random
from playsound import playsound
from Chatbot import neironka
from fuzzywuzzy import fuzz
from numru import numtotext
import vosk
import sounddevice as sd
from playsound import playsound
import queue
cmd_list = {
	"time": ("время", "cколько времени", "который час"),
	"browser": ("браузер", "открой браузер"),
	"hentai": ("ямате кудасай", "хентай"),
	"mkdir": ("создай папку", "папку"),
	"telegram": ("открой телеграм", "телега"),
	"discord": ("открой дискорд", "дискорд"),
	"restart": ("перезагрузка", "рестарт"),
	"shutdown": ("умри", "отключись")
}
def filter(raw_voice: str): 
	cmd = raw_voice.replace("сиеста", "")
	return cmd

def recog(cmd: str):
	rc = {'cmd': '', 'percent': 25}
	for i, v in cmd_list.items():
		for c in v:
			brt = fuzz.ratio(cmd, c)
			if brt > rc['percent']:
				rc['cmd'] = i 
				rc['percent'] = brt
	return rc

def execute(cmd: str):
	match cmd:
		case "time": say(what_time())
		case "browser": webbrowser.open("https://google.com")
		case "hentai": say("Ямате кудасай, няяя")
		case "mkdir": os.mkdir("Project")
		case "telegram": subprocess.Popen('telegram-desktop')
		case "discord": subprocess.Popen('discord')
		case "restart": restart()
		case "shutdown": raise SystemExit

def say(example_text):
	audio_paths = model_torch.save_wav(text=example_text,
                                 speaker=speaker,
                                 sample_rate=sample_rate)
	playsound("test.wav")
	os.remove("test.wav")

def callback(indate, frames, time, status):
	if status:
		print(status, file=sys.stderr)
	q.put(bytes(indate))

def listening():
	with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device, dtype="int16", channels=1, callback=callback):
		rec = vosk.KaldiRecognizer(model, samplerate)
		while True:
			data = q.get()
			if rec.AcceptWaveform(data):
				voice = rec.Result()
				if "сиеста" in voice:
					cmd = recog(filter(voice))
					if cmd['cmd'] not in cmd_list.keys():
						sentence = neironka(filter(voice))
						say(sentence)
					else: 
						execute(cmd['cmd'])



def restart():
	say("Секунду")
	python = sys.executable
	os.execl(python, python, * sys.argv)
	
if __name__ == "__main__":
	device_torch = torch.device('cpu')
	torch.set_num_threads(4)
	local_file = 'model.pt'
	torch.backends.quantized.engine = 'qnnpack'
	if not os.path.isfile(local_file):
	    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/ru_v3.pt',
	                                   local_file)  

	model_torch = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
	model_torch.to(device_torch)

	sample_rate = 48000
	speaker='baya'
	model = vosk.Model("model-small")
	samplerate = 16000 
	device = 2
	q = queue.Queue()
	say("Сиеста начала свою работу, жду указаний, Ня")
	listening()

