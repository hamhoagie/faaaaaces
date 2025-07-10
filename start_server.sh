#!/bin/bash

echo "🚀 Starting FAAAAACES Server..."

# Activate virtual environment (Python 3.12 with DeepFace support)
source .venv/bin/activate

# Set environment variables
export FLASK_ENV=development
export FLASK_DEBUG=True

# Start the server in the background
nohup python3 run_simple.py > server.log 2>&1 &

# Get the process ID
SERVER_PID=$!

echo "✅ Server started with PID: $SERVER_PID"
echo "📄 Logs: tail -f server.log"
echo "🌐 URL: http://localhost:5005"
echo "🛑 Stop: kill $SERVER_PID"

# Save PID for easy stopping
echo $SERVER_PID > server.pid

# Wait a moment and check if server started
sleep 3

if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server is running successfully!"
    echo ""
    echo "🌟 FAAAAACES is ready!"
    echo "🌐 Open your browser to: http://localhost:5005"
    echo ""
    echo "To stop the server: ./stop_server.sh"
else
    echo "❌ Server failed to start. Check server.log for details."
fi