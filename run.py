# run.py
from waitress import serve
from app import app
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Flask app with Waitres")
    serve(app, host='0.0.0.0', port=5000)