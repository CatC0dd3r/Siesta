from core import Core
from colorama import Fore
if __name__ == "__main__":
	core = Core()
	core.model()
	core.say("Сиеста начала свою работу, жду указаний, Ня")
	print(f"{Fore.GREEN}Сиеста слушает")
	while True:
		voice = core.listening()
		if voice:
			voice_input = voice.replace("сиеста", "").strip()
			voice_input_parts = voice_input.split(" ")
			RC = {"cmd": "", "argument": ""}
			voice_input_parts = voice_input.split(" ")
			if len(voice_input_parts) == 1:
				intent = core.neironka(voice_input)
				if intent:
					RC["cmd"] = intent
				else:
					RC["cmd"] = "Не поняла"
			
			if len(voice_input_parts) > 1:
				for guess in range(len(voice_input_parts)):
					intent = core.neironka((" ".join(voice_input_parts[0:guess])).strip())
					if intent:
						RC["argument"] = [voice_input_parts[guess:len(voice_input_parts)]]
						RC["cmd"] = intent

			core.execute(RC)