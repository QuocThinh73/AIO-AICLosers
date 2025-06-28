from googletrans import Translator

def translate_text(text, src='vi', dest='en'):
    """
    Translate text from Vietnamese to English using Google Translate.
    
    Args:
        text (str): Text to translate
        src (str): Source language code (default: 'vi' for Vietnamese)
        dest (str): Destination language code (default: 'en' for English)
    
    Returns:
        str: Translated text
    """
    translator = Translator()
    result = translator.translate(text, dest=dest, src=src)
    return result.text





