# Simple translate fallback to avoid googletrans dependency issues

def translate_text(text, src='vi', dest='en'):
    """
    Simple translate function fallback (bypasses googletrans dependency).
    Returns original text since translation is not critical for core functionality.
    
    Args:
        text (str): Text to translate
        src (str): Source language code (default: 'vi' for Vietnamese)
        dest (str): Destination language code (default: 'en' for English)
    
    Returns:
        str: Original text (translation disabled)
    """
    # Return empty string if input is empty
    if not text or not text.strip():
        return ''
    
    # For now, just return original text
    # TODO: Implement proper translation when googletrans dependency is fixed
    print(f"[INFO] Translation disabled - returning original text: {text[:50]}...")
    return text
