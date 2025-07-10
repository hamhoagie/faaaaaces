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
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )