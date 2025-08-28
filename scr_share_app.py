import os
import logging
import ollama
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import cv2
import numpy as np
import base64
from typing import Dict, Optional
import datetime
from pathlib import Path
from dotenv import load_dotenv
import threading
import queue
import re
import ast
import pytesseract

# Configure Tesseract path if needed (especially on Windows)
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)

# Basic CORS configuration
CORS(app)

# Simple SocketIO setup
socketio = SocketIO(app, async_mode=None)  # Let SocketIO choose the best async mode

# Initialize global variables
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Queue for processing frames
frame_queue = queue.Queue(maxsize=10)
analysis_results = {}

class StreamAnalyzer:
    def __init__(self):
        self.current_analysis: Optional[Dict] = None
        self.last_analysis_time = 0
        self.analysis_interval = 2  # Analyze every 2 seconds
        
    def analyze_code(self, text: str) -> Dict:
        """Analyze code content for errors and provide suggestions"""
        try:
            # Try to parse as Python code to detect syntax errors
            try:
                ast.parse(text)
                has_syntax_error = False
                syntax_error = None
            except SyntaxError as e:
                has_syntax_error = True
                syntax_error = str(e)
            
            # Look for common error patterns
            error_patterns = {
                "undefined variable": r"NameError.*?'(\w+)' is not defined",
                "type error": r"TypeError:.*",
                "index error": r"IndexError:.*",
                "key error": r"KeyError:.*",
                "import error": r"ImportError:.*",
                "indentation error": r"IndentationError:.*"
            }
            
            found_errors = []
            for error_type, pattern in error_patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    found_errors.append({
                        "type": error_type,
                        "details": matches
                    })
            
            return {
                "success": True,
                "has_syntax_error": has_syntax_error,
                "syntax_error": syntax_error,
                "other_errors": found_errors,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def analyze_webpage(self, text: str) -> Dict:
        """Analyze webpage content and provide summary"""
        try:
            # Extract potential headings
            heading_pattern = r'[A-Z][A-Za-z\s]{2,50}(?:\n|$)'
            headings = re.findall(heading_pattern, text)
            
            # Extract potential links
            link_pattern = r'https?://\S+'
            links = re.findall(link_pattern, text)
            
            # Get text summary using Ollama
            summary_prompt = f"Summarize this webpage content in 3-4 key points:\n\n{text[:2000]}"
            summary_response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a web content analyzer. Provide concise, accurate summaries."},
                    {"role": "user", "content": summary_prompt}
                ]
            )
            
            summary = summary_response.get("message", {}).get("content", "")
            
            return {
                "success": True,
                "headings": headings[:10],
                "links": links[:10],
                "summary": summary,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def process_frame(self, frame_data: str, room_id: str) -> None:
        """Process a single frame from the stream"""
        try:
            current_time = datetime.datetime.now().timestamp()
            
            # Only analyze every few seconds to avoid overload
            if current_time - self.last_analysis_time < self.analysis_interval:
                return
                
            self.last_analysis_time = current_time
            
            # Decode base64 frame
            try:
                encoded_data = frame_data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Error decoding frame: {e}")
                return
            
            if frame is None:
                logger.error("Failed to decode frame")
                return
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Resize if the image is too large (helps with OCR accuracy)
            max_dimension = 1920
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / max(width, height)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing for better OCR
            # 1. Denoise
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 2. Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Thresholding
            _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Configure Tesseract parameters
            custom_config = r'--oem 3 --psm 6'  # Assume uniform text block
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(
                threshold,
                config=custom_config,
                lang='eng'  # Use English language
            )
            
            # Clean up the extracted text
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = text.replace('\n\n', '\n')  # Remove extra newlines
            
            if text:
                logger.info(f"Successfully extracted text from frame: {text[:100]}...")
                # Store analysis results for this room
                analysis_results[room_id] = {
                    "text": text,
                    "timestamp": current_time,
                    "frame_size": f"{width}x{height}"
                }
            else:
                logger.warning("No text extracted from frame")
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")

# Initialize stream analyzer
stream_analyzer = StreamAnalyzer()

def process_frames():
    """Background worker to process frames from the queue"""
    while True:
        try:
            frame_data, room_id = frame_queue.get()
            if frame_data is None:
                break
            stream_analyzer.process_frame(frame_data, room_id)
            frame_queue.task_done()
        except Exception as e:
            logger.error(f"Error in frame processing worker: {str(e)}")

# Start frame processing worker
frame_processor = threading.Thread(target=process_frames, daemon=True)
frame_processor.start()

@app.route('/')
def index():
    """Serve the main application page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return jsonify({
            'error': 'Failed to load application',
            'details': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.datetime.now().isoformat(),
            'config': {
                'host': os.environ.get('HOST', '0.0.0.0'),
                'port': os.environ.get('PORT', 9092),
                'ssl': os.path.exists('cert.pem') and os.path.exists('key.pem'),
                'debug': os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join')
def handle_join(data):
    """Handle client joining a room"""
    room_id = data.get('room')
    if room_id:
        join_room(room_id)
        emit('joined', {'room': room_id})

@socketio.on('leave')
def handle_leave(data):
    """Handle client leaving a room"""
    room_id = data.get('room')
    if room_id:
        leave_room(room_id)
        emit('left', {'room': room_id})

@socketio.on('stream-frame')
def handle_frame(data):
    """Handle incoming frame from screen share stream"""
    try:
        frame_data = data.get('frame')
        room_id = data.get('room')
        
        if frame_data and room_id:
            logger.info(f"Received frame from room {room_id}")
            
            # Add frame to processing queue
            try:
                frame_queue.put_nowait((frame_data, room_id))
                logger.info("Frame added to processing queue")
            except queue.Full:
                logger.warning("Frame queue full, skipping frame")
        else:
            logger.warning(f"Invalid frame data received: frame_data={bool(frame_data)}, room_id={bool(room_id)}")
                
    except Exception as e:
        logger.error(f"Error handling frame: {str(e)}")

@socketio.on('query')
def handle_query(data):
    """Handle user query about the screen content"""
    try:
        query = data.get('query', '').strip()
        room_id = data.get('room')
        
        if not query or not room_id:
            emit('response', {'error': 'Invalid query or room ID'})
            return
            
        # Get latest analysis results for this room
        result = analysis_results.get(room_id)
        if not result:
            emit('response', {
                'error': 'Waiting for screen content. Please make sure:\n' +
                        '1. You have started screen sharing\n' +
                        '2. There is visible text on your screen\n' +
                        '3. The shared window/tab is active'
            })
            return
            
        # Validate the extracted text
        extracted_text = result.get('text', '').strip()
        if not extracted_text:
            emit('response', {
                'error': 'No text could be extracted from your screen. Please ensure:\n' +
                        '1. There is visible text content\n' +
                        '2. The text is clear and readable\n' +
                        '3. The screen sharing quality is good'
            })
            return
            
        extracted_text = result.get('text', '')
        
        # Determine analysis type based on content and query
        analysis_type = "general"
        if any(kw in query.lower() for kw in ['code', 'error', 'bug', 'fix', 'syntax']):
            analysis_type = "code"
        elif any(kw in query.lower() for kw in ['website', 'webpage', 'site', 'url', 'link', 'summary']):
            analysis_type = "webpage"
            
        # Perform specific analysis
        if analysis_type == "code":
            code_analysis = stream_analyzer.analyze_code(extracted_text)
            if not code_analysis["success"]:
                emit('response', {'error': f'Code analysis failed: {code_analysis["error"]}'})
                return
                
            prompt = f"""Analyze this code and user's question. Provide specific, helpful suggestions.

Code content:
{extracted_text}

Analysis results:
- Syntax errors: {"Yes - " + code_analysis["syntax_error"] if code_analysis["has_syntax_error"] else "None"}
- Other errors: {code_analysis["other_errors"]}

User question:
{query}

Provide a detailed response addressing the user's specific question and any detected issues."""
            
        elif analysis_type == "webpage":
            web_analysis = stream_analyzer.analyze_webpage(extracted_text)
            if not web_analysis["success"]:
                emit('response', {'error': f'Webpage analysis failed: {web_analysis["error"]}'})
                return
                
            prompt = f"""Analyze this webpage content and user's question. Provide a clear, informative response.

Webpage content summary:
{web_analysis["summary"]}

Key sections found:
{chr(10).join(web_analysis["headings"][:5])}

User question:
{query}

Provide a detailed response addressing the user's specific question and summarizing the relevant webpage content."""
            
        else:
            prompt = f"""Analyze this screen content and user's question. Provide a helpful response.

Screen content:
{extracted_text}

User question:
{query}

Provide a detailed response addressing the user's specific question about the screen content."""
        
        # Get response from Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a screen analysis assistant. Provide clear, specific answers to user questions about their screen content."},
                {"role": "user", "content": prompt}
            ]
        )
        
        assistant_response = response.get("message", {}).get("content", "")
        
        emit('response', {
            'response': assistant_response,
            'analysis_type': analysis_type
        })
        
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        emit('response', {'error': str(e)})

def create_self_signed_cert():
    """Create a self-signed certificate if none exists"""
    if not os.path.exists('cert.pem') or not os.path.exists('key.pem'):
        try:
            import subprocess
            subprocess.run([
                'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
                '-keyout', 'key.pem', '-out', 'cert.pem',
                '-days', '365', '-nodes',
                '-subj', '/CN=localhost'
            ], check=True)
            logger.info("Created self-signed certificate")
            return True
        except Exception as e:
            logger.error(f"Failed to create self-signed certificate: {e}")
            return False
    return True

if __name__ == '__main__':
    try:
        # Use port 5000 which is typically allowed by most firewalls
        port = 5000
        print(f"\n{'='*50}")
        print(f"Server starting at: http://localhost:{port}")
        print(f"{'='*50}\n")
        
        socketio.run(app, host='localhost', port=port, debug=True)
    except Exception as e:
        print(f"\nError starting server: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure port 5000 is not in use")
        print("2. Try running as administrator/root")
        print("3. Check firewall settings")
        print("4. If using a VPN, try disconnecting it\n")