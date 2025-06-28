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
    # Return empty string if input is empty
    if not text or not text.strip():
        return ''
    
    translator = Translator()
    result = translator.translate(text, dest=dest, src=src)
    return result.text if result and result.text else ''





