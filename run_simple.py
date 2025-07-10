#!/usr/bin/env python3
"""
Simplified FAAAAACES runner for testing
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask
    print("✅ Flask imported successfully")
    
    # Create minimal app first
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-secret-key'
    
    @app.route('/')
    def home():
        return '''
        <h1>🎭 FAAAAACES - Face Recognition App</h1>
        <h2>✅ Basic server is running!</h2>
        <p>🔧 Full features loading...</p>
        <hr>
        <h3>🎯 Planned Features:</h3>
        <ul>
            <li>📤 Video file upload</li>
            <li>🔗 Instagram & TikTok URL processing</li>
            <li>👥 Face detection and clustering</li>
            <li>📊 Dashboard and analytics</li>
        </ul>
        <p><a href="/status">Check App Status</a></p>
        '''
    
    @app.route('/status')
    def status():
        try:
            # Test imports
            import cv2
            opencv_version = cv2.__version__
            opencv_status = "✅ Available"
        except:
            opencv_version = "Not installed"
            opencv_status = "❌ Missing"
            
        try:
            import yt_dlp
            ytdlp_status = "✅ Available"
        except:
            ytdlp_status = "❌ Missing"
            
        try:
            from app.models.database import init_db
            db_status = "✅ Available"
        except Exception as e:
            db_status = f"❌ Error: {str(e)}"
            
        return f'''
        <h2>🔍 FAAAAACES System Status</h2>
        <ul>
            <li><strong>OpenCV:</strong> {opencv_status} (v{opencv_version})</li>
            <li><strong>yt-dlp:</strong> {ytdlp_status}</li>
            <li><strong>Database:</strong> {db_status}</li>
        </ul>
        <p><a href="/">Back to Home</a></p>
        '''
    
    print("✅ Routes configured")
    
    # Try to import full app components
    try:
        from app import create_app
        print("✅ Full app components available")
        full_app = create_app()
        print("✅ Full app created successfully")
        app = full_app  # Use full app if available
    except Exception as e:
        print(f"⚠️  Using minimal app due to: {e}")
    
    if __name__ == '__main__':
        port = 5005
        print(f"\n🚀 FAAAAACES starting on port {port}")
        print(f"🌐 Open: http://localhost:{port}")
        print(f"🌐 Or:   http://127.0.0.1:{port}")
        print("\n📱 Ready to test face recognition!")
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True,
            use_reloader=False  # Disable reloader to avoid double startup
        )
        
except Exception as e:
    print(f"❌ Error starting app: {e}")
    import traceback
    traceback.print_exc()