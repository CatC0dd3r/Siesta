import torch
import json
from playsound import playsound
import sys
import os
import vosk
import queue
import sounddevice as sd
from numru import numtotext
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skills import *
class Core:
	def __init__(self):
		self.device_torch = torch.device('cpu')
		self.local_file = 'model.pt'
		if not os.path.isfile(self.local_file):
			torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/ru_v3.pt', self.local_file)  
		
		torch.backends.quantized.engine = 'qnnpack'
		self.model_tts = torch.package.PackageImporter(self.local_file).load_pickle("tts_models", "model")
		self.model_tts.to(self.device_torch)

		self.sample_rate = 48000
		self.speaker='baya'
		self.model_stt = vosk.Model("model-small")
		self.samplerate = 16000 
		self.device_stt = 1
		self.q = queue.Queue()
		self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
		self.classifier_probability = LogisticRegression()
		self.classifier = LinearSVC()
		self.cmd = {
		    "intents": {
		        "greeting": {
		            "patterns": ["привет", "здравствуй"]
		        },
		        "bye": {
		            "patterns": ["пока", "до свидания"]
		        },
		        "google_search": {
		            "patterns": ["найди", "поищи"]
		        },
		        "off": {
		        	"patterns": ["отключись", "умри"]
		        },
		        "reboot": {
		        	"patterns": ["рестарт", "перезапуск"]
		        },
		        "time": {
		        	"patterns": ["который час", "время"]
		        }
		    },
		    "not_found_command": "idk"
		}

	def model(self):
		corpus = []
		target_vector = []
		for intent_name, intent_data in self.cmd["intents"].items():
			for pattern in intent_data["patterns"]:
				corpus.append(pattern)
				target_vector.append(intent_name)

		training_vector = self.vectorizer.fit_transform(corpus)
		self.classifier_probability.fit(training_vector, target_vector)
		self.classifier.fit(training_vector, target_vector)


	def neironka(self, request):
		best_intent = self.classifier.predict(self.vectorizer.transform([request]))[0]

		index_of_best_intent = list(self.classifier_probability.classes_).index(best_intent)
		probabilities = self.classifier_probability.predict_proba(self.vectorizer.transform([request]))[0]

		best_intent_probability = probabilities[index_of_best_intent]
		print(best_intent_probability)
		if best_intent_probability > 0.300:
			return best_intent
	

	def say(self, text):
		audio_paths = self.model_tts.save_wav(text=text,
                                 speaker=self.speaker,
                                 sample_rate=self.sample_rate)
		playsound("test.wav")
		os.remove("test.wav")

	def listening(self):
		def callback(indate, frames, time, status):
			if status:
				print(status, file=sys.stderr)
			self.q.put(bytes(indate))	
		with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000, device=self.device_stt, dtype="int16", channels=1, callback=callback):
			rec = vosk.KaldiRecognizer(self.model_stt, self.samplerate)
			while True:
				data = self.q.get()
				if rec.AcceptWaveform(data):
					voice = rec.Result()
					voice = json.loads(voice)
					voice = voice["text"]
					if "сиеста" in voice:
						return voice
						
	def execute(self, cmd: dict):
		match cmd["cmd"]:
			case "greeting": self.say("Приветья")
			case "bye": self.say("Пока(")
			case "google_search": skill_search_google.search_google(*cmd["argument"])
			case "off": raise SystemExit
			case "reboot": python = sys.executable; os.execl(python, python, * sys.argv)
			case "time": now = datetime.datetime.now(); self.say("Сейчас " + numtotext(now.hour) + ' и' + numtotext(now.minute))
			case _: self.say("Не поняла")
