import json
from io import BytesIO
from PIL import Image
from flask import request
from app.translate import translate_text


def parse_image_upload():
    """
    Parse uploaded image from request files.
    
    Returns:
        PIL.Image or None: Processed image object or None if no image
    """
    if 'file' not in request.files or request.files['file'].filename == '':
        return None
    
    file = request.files['file']
    file.seek(0)
    image_data = file.read()
    image_stream = BytesIO(image_data)
    return Image.open(image_stream).convert('RGB')


def parse_search_params():
    """
    Parse search parameters from FormData.
    
    Returns:
        dict: Parsed search parameters
    """
    # Get basic parameters
    query = translate_text(request.form.get('query', ''))
    ocr = request.form.get('ocr_text', '')
    topK = int(request.form.get('topK', 100))
    
    # Parse JSON arrays
    models = json.loads(request.form.get('models', '[]'))
    objects = json.loads(request.form.get('objects', '[]'))
    
    return {
        'query': query,
        'ocr_text': ocr,
        'models': models,
        'objects': objects,
        'topK': topK
    }


def parse_search_request():
    """
    Parse complete search request including image and parameters.
    
    Returns:
        tuple: (uploaded_image, search_params)
    """
    uploaded_image = parse_image_upload()
    search_params = parse_search_params()
    
    return uploaded_image, search_params 