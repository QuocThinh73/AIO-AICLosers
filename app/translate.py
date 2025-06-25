from googletrans import Translator

def translate_text(text, src='vi', dest='en'):
    translator = Translator()
    result = translator.translate(text, dest=dest, src=src)
    return result.text





