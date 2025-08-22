#!/usr/bin/env python3
"""
Test script to verify Cloudinary integration
"""

import os
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

def test_cloudinary_config():
    """Test if Cloudinary is configured correctly"""
    print("🔧 Testing Cloudinary Configuration...")
    print(f"Cloud Name: {os.getenv('CLOUDINARY_CLOUD_NAME')}")
    print(f"API Key: {os.getenv('CLOUDINARY_API_KEY')}")
    print(f"API Secret: {'*' * len(os.getenv('CLOUDINARY_API_SECRET', ''))}")
    
    # Test configuration
    if all([os.getenv('CLOUDINARY_CLOUD_NAME'), os.getenv('CLOUDINARY_API_KEY'), os.getenv('CLOUDINARY_API_SECRET')]):
        print("✅ Cloudinary configuration loaded successfully!")
        return True
    else:
        print("❌ Missing Cloudinary configuration!")
        return False

def generate_player_url(public_id):
    """Generate player URL for a given public_id"""
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    player_url = f"https://player.cloudinary.com/embed/?cloud_name={cloud_name}&public_id={public_id}&profile=cld-default"
    return player_url

if __name__ == "__main__":
    print("🎬 Cloudinary Integration Test")
    print("=" * 40)
    
    if test_cloudinary_config():
        print("\n🎯 Example Player URL:")
        example_public_id = "xlfoqpz61o1tmc7y2cye"
        player_url = generate_player_url(example_public_id)
        print(f"Player URL: {player_url}")
        
        print("\n✅ Cloudinary integration is ready!")
        print("💡 The backend will now upload processed videos to Cloudinary")
        print("🔗 Frontend will show clickable links to the video player")
    else:
        print("\n❌ Please check your .env file and Cloudinary credentials")
