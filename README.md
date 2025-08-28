# Screen Share Assistant

A real-time screen sharing application with AI-powered content analysis. This application allows users to share their screen and get intelligent analysis and answers about the content being shared.

## Features

- Real-time screen sharing using WebRTC
- Live text extraction from screen content using OCR
- AI-powered content analysis and question answering
- Support for code analysis and error detection
- Web page content summarization
- Real-time chat interface

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Tesseract OCR
- Ollama (for AI analysis)

### Installing Tesseract OCR

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

#### macOS
```bash
brew install tesseract
```

#### Windows
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd screenshare_agent.py
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
# Create .env file
echo "OLLAMA_MODEL=gemma3:12b" > .env
```

## Usage

1. Start the server:
```bash
python scr_share_app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the interface:
   - Click "Start Screen Share" to begin sharing your screen
   - Select the window or tab you want to share
   - Use the chat interface to ask questions about the content
   - The AI will analyze the screen content and provide responses

## Features in Detail

### Screen Sharing
- Real-time screen capture
- Support for multiple screen sharing sessions
- Automatic quality adjustment for optimal performance

### Text Extraction
- OCR-based text extraction from screen content
- Image preprocessing for better text recognition
- Support for various content types (code, web pages, documents)

### AI Analysis
- Code error detection and suggestions
- Web page content summarization
- Context-aware question answering
- Real-time content analysis

### Chat Interface
- Real-time communication
- Error handling and status updates
- Support for various query types
- Clear feedback on analysis status

## Troubleshooting

### Common Issues

1. Connection Issues
   - Make sure no other application is using port 5000
   - Check firewall settings
   - Try running with a different port: `PORT=8080 python scr_share_app.py`

2. Screen Sharing Issues
   - Use a modern browser (Chrome, Firefox, Edge)
   - Make sure you're on HTTP localhost or HTTPS
   - Grant necessary permissions when prompted

3. Text Recognition Issues
   - Ensure text is clearly visible
   - Try adjusting screen zoom level
   - Make sure the shared window is in focus

### Debug Mode

To run in debug mode with additional logging:
```bash
export FLASK_DEBUG=true
python scr_share_app.py
```

## API Reference

### WebSocket Events

- `connect`: Client connection event
- `disconnect`: Client disconnection event
- `join`: Join a room for screen sharing
- `stream-frame`: Send screen frame for analysis
- `query`: Send question about screen content

### HTTP Endpoints

- `GET /`: Main application interface
- `GET /health`: Health check endpoint
- `GET /static/<path>`: Static file serving

## Dependencies

- Flask: Web framework
- Flask-SocketIO: WebSocket support
- OpenCV: Image processing
- Tesseract: OCR engine
- Ollama: AI model integration
- Additional dependencies in requirements.txt

## License

[Your License Here]

## Contributing

[Contribution Guidelines]

## Security

- Screen sharing requires explicit user permission
- Data is processed locally
- No permanent storage of screen content
- Secure WebSocket communication
