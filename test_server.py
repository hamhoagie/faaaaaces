#!/usr/bin/env python3
"""
Simple test server to check Flask connectivity
"""
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <h1>🎉 FAAAAACES Test Server</h1>
    <p>✅ Flask is working!</p>
    <p>✅ Connection successful!</p>
    <p><a href="/test">Test Page</a></p>
    '''

@app.route('/test')
def test():
    return '<h2>✅ Test page works!</h2><p><a href="/">Back to Home</a></p>'

if __name__ == '__main__':
    print("🚀 Starting test server...")
    print("📱 Visit: http://localhost:5004")
    print("🌐 Or try: http://127.0.0.1:5004")
    app.run(host='0.0.0.0', port=5004, debug=True)