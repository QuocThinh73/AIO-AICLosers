"""
Flask application for image search API
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
