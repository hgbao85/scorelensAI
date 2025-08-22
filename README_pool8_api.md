# Pool8 AI Backend & Frontend System

This system converts the `pool8.py` script into a web-based API service with a frontend interface.

## 📁 Files Created

1. **`backend_server.py`** - Flask backend server that processes videos using pool8.py
2. **`frontend.html`** - Web interface for uploading videos and viewing results
3. **`requirements.txt`** - Updated with Flask dependencies

## 🚀 Quick Start

### 1. Install Dependencies

Make sure you're in your virtual environment, then install the required packages:

```bash
# Activate your virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
python backend_server.py
```

The server will:
- Check that model files exist (`train18/weights/best.pt` and `best_cnn_model.pth`)
- Import and validate `pool8.py`
- Start the Flask server on `http://localhost:5000`

### 3. Open the Frontend

Open `frontend.html` in your web browser or navigate to the file location.

## 🎯 How It Works

### Backend API

The backend server provides these endpoints:

- **`GET /health`** - Check server status and model availability
- **`POST /process_video`** - Upload and process a video file
- **`GET /`** - API information

### Video Processing Flow

1. **Upload**: Frontend sends video file to `/process_video` endpoint
2. **Process**: Backend calls `pool8.py` main function with the uploaded video
3. **Return**: Backend returns processed video (base64) and CSV events
4. **Display**: Frontend shows results and provides download links

### Input/Output

**Input:**
- Video file (MP4, MOV, AVI, etc.)
- Maximum file size: 500MB

**Output:**
- Processed video with ball tracking and game state overlays
- CSV file with game events (collisions, turns, ball pocketing, etc.)

## 🌐 Frontend Features

- **Drag & Drop Upload**: Easy video file uploading
- **Progress Indication**: Shows processing status
- **Video Player**: Plays processed video with controls
- **Event Table**: Displays game events in a sortable table
- **Download Options**: Download processed video and CSV files
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Shows helpful error messages

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cpu"
}
```

### Process Video
```bash
curl -X POST -F "video=@your_video.mp4" http://localhost:5000/process_video
```

Response:
```json
{
  "success": true,
  "message": "Video processed successfully",
  "processed_video": "base64_encoded_video_data",
  "events_csv": "Event_Type,Turn,Frame,Timestamp,Details\n...",
  "filename": "your_video.mp4"
}
```

## 🛠️ Technical Details

### Backend Architecture
- **Flask Web Server**: Handles HTTP requests and file uploads
- **pool8.py Integration**: Uses the original pool8.py main function directly
- **Temporary Files**: Safely handles file uploads and cleanup
- **Base64 Encoding**: Returns processed video as base64 for web display

### Frontend Architecture
- **Pure HTML/CSS/JavaScript**: No external dependencies
- **Modern UI**: Clean, responsive design with animations
- **AJAX Communication**: Communicates with backend via fetch API
- **File Handling**: Supports drag & drop and traditional file selection

## 🔍 Troubleshooting

### Backend Issues
- **Models not loading**: Check that model files exist in correct paths
- **Import errors**: Ensure all dependencies are installed in virtual environment
- **Memory issues**: Large videos may require significant RAM

### Frontend Issues
- **Cannot connect to backend**: Ensure backend server is running on port 5000
- **CORS errors**: Try serving the HTML file through a local web server
- **Large file uploads**: Check that video file size is under 500MB limit

### Common Solutions

1. **Virtual Environment**: Make sure you're using the virtual environment:
   ```bash
   venv\Scripts\activate  # Windows
   python backend_server.py
   ```

2. **Dependencies**: Install missing packages:
   ```bash
   pip install flask werkzeug
   ```

3. **Model Files**: Ensure these files exist:
   - `train18/weights/best.pt`
   - `best_cnn_model.pth`

## 📊 Performance Tips

- **GPU Acceleration**: Use CUDA-enabled PyTorch for faster processing
- **Video Resolution**: Lower resolution videos process faster
- **File Format**: MP4 format is recommended for best compatibility

## 🔒 Security Notes

- This is a development setup - do not expose to the internet without proper security measures
- File uploads are limited to 500MB to prevent abuse
- Temporary files are cleaned up after processing
- Consider adding authentication for production use

## 🎯 Usage Example

1. **Start the server**:
   ```bash
   python backend_server.py
   ```

2. **Open frontend**: Open `frontend.html` in your browser

3. **Upload video**: Drag and drop a pool game video

4. **Process**: Click "Process Video" and wait for results

5. **Download**: Get the processed video and events CSV

The system provides the same functionality as `pool8.py` but through a user-friendly web interface!
