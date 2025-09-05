from googletrans import Translator


async def translate_text(text: str, src_lang: str, dest_lang: str) -> str:
    if not text or not text.strip():
        return ""
    
    async with Translator() as translator:
        result = await translator.translate(text, dest=dest_lang, src=src_lang)
        
    return result.text if result and result.text else ''