#!/usr/bin/env python3
"""
FAAAAACES - Face Recognition Video Processing Application
"""
import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('faces', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    
    # Run the application
    port = int(os.getenv('PORT', 5001))  # Use 5001 as default to avoid conflicts
    print(f"\n🚀 Starting FAAAAACES on http://localhost:{port}")
    print("📹 Ready to process videos with face recognition!")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )