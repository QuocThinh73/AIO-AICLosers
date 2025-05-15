class Translation:
    """Translation class for translating text from Vietnamese to English.
    
    Supports multiple translation backends.
    """
    
    def __init__(self, backend: str = "google"):
        """Initialize the translator.
        
        Args:
            backend (str): Translation backend to use ("google", "huggingface", etc.)
        """
        self.backend = backend
        
        # Initialize the appropriate backend
        if backend == "google":
            try:
                from googletrans import Translator
                self.translator = Translator()
                self.translate_func = self._google_translate
            except ImportError:
                print("Warning: googletrans not installed.")
                raise ImportError("Please install googletrans: pip install googletrans==4.0.0-rc1")
        elif backend == "huggingface":
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                model_name = "VietAI/envit5-translation"
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.translate_func = self._huggingface_translate
            except ImportError:
                print("Warning: transformers not installed.")
                raise ImportError("Please install transformers: pip install transformers")
        else:
            raise ValueError(f"Unsupported translation backend: {backend}")
    
    def __call__(self, text: str) -> str:
        """Translate text from Vietnamese to English.
        
        Args:
            text (str): Text to translate (in Vietnamese)
            
        Returns:
            str: Translated text (in English)
        """
        translation = self.translate_func(text)

        return translation
    
    def _google_translate(self, text: str) -> str:
        """Translate using Google Translate API.
        
        Args:
            text (str): Text to translate (in Vietnamese)
            
        Returns:
            str: Translated text (in English)
        """
        try:
            return self.translator.translate(text, src='vi', dest='en').text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def _huggingface_translate(self, text: str) -> str:
        """Translate using HuggingFace model.
        
        Args:
            text (str): Text to translate (in Vietnamese)
            
        Returns:
            str: Translated text (in English)
        """
        import torch
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]