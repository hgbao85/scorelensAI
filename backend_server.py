#!/usr/bin/env python3
"""
Pool8 AI Backend Server
Converts pool8.py into a REST API service that processes videos and returns results.
"""

import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# ===== FLASK APP =====
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size

# Global variables for models
models_loaded = False

def load_models():
    """Load models by importing pool8.py and carom.py and checking dependencies"""
    global models_loaded
    try:
        # Check if model files exist
        pool8_yolo_path = "train18/weights/best.pt"
        pool8_cnn_path = "best_cnn_model.pth"
        carom_yolo_path = "train15/weights/best.pt"

        # Check Pool8 models
        if not os.path.exists(pool8_yolo_path):
            print(f"❌ Pool8 YOLO model not found: {pool8_yolo_path}")
            return False

        if not os.path.exists(pool8_cnn_path):
            print(f"❌ Pool8 CNN model not found: {pool8_cnn_path}")
            return False

        # Check Carom model
        if not os.path.exists(carom_yolo_path):
            print(f"❌ Carom YOLO model not found: {carom_yolo_path}")
            return False

        # Try importing both modules
        import pool8
        import carom
        print("✅ pool8.py imported successfully")
        print("✅ carom.py imported successfully")
        print(f"✅ Pool8 YOLO model found: {pool8_yolo_path}")
        print(f"✅ Pool8 CNN model found: {pool8_cnn_path}")
        print(f"✅ Carom YOLO model found: {carom_yolo_path}")

        models_loaded = True
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def upload_video_to_cloudinary(video_path, original_filename):
    """
    Upload video to Cloudinary and return player URL
    """
    try:
        print("☁️ Uploading video to Cloudinary...")

        # Upload video to Cloudinary
        response = cloudinary.uploader.upload(
            video_path,
            resource_type="video",
            public_id=f"processed_{original_filename}_{int(os.path.getmtime(video_path))}"
        )

        public_id = response['public_id']
        cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')

        # Generate player URL
        player_url = f"https://player.cloudinary.com/embed/?cloud_name={cloud_name}&public_id={public_id}&profile=cld-default"

        print(f"✅ Video uploaded successfully. Public ID: {public_id}")
        return {
            "success": True,
            "player_url": player_url,
            "public_id": public_id,
            "cloudinary_url": response.get('secure_url', response.get('url'))
        }

    except Exception as e:
        print(f"❌ Error uploading to Cloudinary: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def process_video_with_pool8(video_path, output_video_path, csv_output_path):
    """
    Process video using pool8.py main function
    """
    try:
        from pool8 import main as pool8_main

        # Call pool8.py main function
        pool8_main(
            video_path=video_path,
            yolo_model_path="train18/weights/best.pt",
            cnn_model_path="best_cnn_model.pth",
            csv_output_path=csv_output_path,
            output_video_path=output_video_path
        )
        return True
    except Exception as e:
        print(f"❌ Error in pool8 processing: {e}")
        return False

def process_video_with_carom(video_path, output_video_path, csv_output_path):
    """
    Process video using carom.py main function
    """
    try:
        from carom import main as carom_main

        # Call carom.py main function
        carom_main(
            video_path=video_path,
            model_path="train15/weights/best.pt",
            csv_output_path=csv_output_path,
            output_video_path=output_video_path
        )
        return True
    except Exception as e:
        print(f"❌ Error in carom processing: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "message": "Pool8 AI Backend Server is running"
    })

@app.route('/process_video', methods=['POST'])
def process_video():
    """Process uploaded video and return processed video + CSV events"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400

    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 500

    # Get analysis type from form data (default to pool8)
    analysis_type = request.form.get('analysis_type', 'pool8').lower()
    if analysis_type not in ['pool8', 'carom']:
        return jsonify({"error": "Invalid analysis type. Must be 'pool8' or 'carom'"}), 400

    try:
        print(f"📹 Processing video: {video_file.filename} with {analysis_type} analysis")

        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
            video_file.save(temp_input.name)
            temp_input_path = temp_input.name

        temp_output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output_video.close()

        temp_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_csv_file.close()

        print(f"🔄 Starting video processing with {analysis_type}.py...")

        # Process video using selected analysis type
        if analysis_type == 'pool8':
            success = process_video_with_pool8(
                temp_input_path,
                temp_output_video.name,
                temp_csv_file.name
            )
        else:  # carom
            success = process_video_with_carom(
                temp_input_path,
                temp_output_video.name,
                temp_csv_file.name
            )

        if not success:
            # Clean up on failure
            os.unlink(temp_input_path)
            os.unlink(temp_output_video.name)
            os.unlink(temp_csv_file.name)
            return jsonify({"error": "Video processing failed"}), 500

        print("✅ Video processing completed")

        # Read CSV content
        with open(temp_csv_file.name, 'r') as f:
            csv_content = f.read()

        # Upload processed video to Cloudinary
        upload_result = upload_video_to_cloudinary(temp_output_video.name, secure_filename(video_file.filename))

        # Clean up temporary files
        os.unlink(temp_input_path)
        os.unlink(temp_output_video.name)
        os.unlink(temp_csv_file.name)

        if not upload_result["success"]:
            return jsonify({
                "error": f"Video processing completed but upload failed: {upload_result['error']}"
            }), 500

        print("📦 Preparing response...")

        return jsonify({
            "success": True,
            "message": f"Video processed successfully with {analysis_type} analysis",
            "player_url": upload_result["player_url"],
            "cloudinary_url": upload_result["cloudinary_url"],
            "public_id": upload_result["public_id"],
            "events_csv": csv_content,
            "filename": secure_filename(video_file.filename),
            "analysis_type": analysis_type
        })

    except Exception as e:
        print(f"❌ Error processing video: {e}")
        # Clean up on error
        try:
            os.unlink(temp_input_path)
            os.unlink(temp_output_video.name)
            os.unlink(temp_csv_file.name)
        except:
            pass
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def index():
    """Serve basic info about the API"""
    return jsonify({
        "name": "Pool8 & Carom AI Backend Server",
        "version": "1.0",
        "description": "Backend service for processing pool and carom game videos with AI",
        "endpoints": {
            "/health": "GET - Check server health",
            "/process_video": "POST - Process video file (multipart/form-data with 'video' field and optional 'analysis_type' field)",
            "/": "GET - This info"
        },
        "analysis_types": {
            "pool8": "Pool game analysis using YOLO + CNN models (train18/weights/best.pt + best_cnn_model.pth)",
            "carom": "Carom game analysis using YOLO model only (train15/weights/best.pt)"
        },
        "models_loaded": models_loaded,
        "max_file_size": "2GB"
    })

if __name__ == '__main__':
    print("🎱 Pool8 & Carom AI Backend Server")
    print("=" * 50)

    # Load models
    print("📦 Loading models...")
    if load_models():
        print("\n🚀 Server ready!")
        print("📍 Server will be available at: http://localhost:5000")
        print("📋 API Endpoints:")
        print("   GET  /health - Check server status")
        print("   POST /process_video - Process video file")
        print("   GET  / - API information")
        print("\n🎯 Analysis Types:")
        print("   pool8 - Pool game analysis (YOLO + CNN)")
        print("   carom - Carom game analysis (YOLO only)")
        print("⏹️  Press Ctrl+C to stop the server\n")

        # Start Flask server
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\n❌ Failed to load models. Server not started.")
        print("💡 Please ensure:")
        print("   - pool8.py and carom.py are in the same directory")
        print("   - Model files exist:")
        print("     * train18/weights/best.pt (Pool8 YOLO)")
        print("     * best_cnn_model.pth (Pool8 CNN)")
        print("     * train15/weights/best.pt (Carom YOLO)")
        print("   - All dependencies are installed")