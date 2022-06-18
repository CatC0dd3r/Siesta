# Siesta V0.0.1
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
	"shutdown": ("умри", "отключись"),
	"shutdown_system": ("отключи систему", "убей систему нахуй"),
	"restart_system": ("перезапуск", "перезагрузи систему")
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
	if cmd == "time":
		now = datetime.datetime.now()
		say("Сейчас " + numtotext(now.hour) + ' и' + numtotext(now.minute))
	if cmd == "browser":
		webbrowser.open("https://google.com")
		say("Открыла")
	if cmd == "hentai":
		say("Ямате кудасай, няяя")
	if cmd == "mkdir":
		os.mkdir("Project")
		say("Папка создана! НЯЯЯЯ")
	if cmd == "telegram":
		subprocess.Popen('telegram-desktop')
		say("Открыла")
	if cmd == "discord":
		subprocess.Popen('discord')
	if cmd == "restart":
		say("Секунду")
		python = sys.executable
		os.execl(python, python, * sys.argv)
	if cmd == "shutdown":
		say("Пока!!))")
		raise SystemExit
	if cmd == "shutdown_system":
		os.system("shutdown")
	if cmd == "reboot":
		say("Перезагружаю систему")
		os.system("reboot")
	if cmd not in cmd_list.keys():
		say(neironka(cmd))

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
					execute(cmd['cmd'])

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

