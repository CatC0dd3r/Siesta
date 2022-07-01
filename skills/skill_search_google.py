import webbrowser
def search_google(search: list):
    if search == None: return
    search = ' '.join(search).strip()
    url = "https://google.com/search?q=" + search
    webbrowser.get().open(url)