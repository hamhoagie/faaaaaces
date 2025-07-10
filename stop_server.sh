#!/bin/bash

echo "🛑 Stopping FAAAAACES Server..."

# Check if PID file exists
if [ -f "server.pid" ]; then
    SERVER_PID=$(cat server.pid)
    
    # Check if process is running
    if ps -p $SERVER_PID > /dev/null; then
        echo "🔄 Stopping server with PID: $SERVER_PID"
        kill $SERVER_PID
        
        # Wait for graceful shutdown
        sleep 2
        
        # Force kill if still running
        if ps -p $SERVER_PID > /dev/null; then
            echo "⚡ Force stopping server..."
            kill -9 $SERVER_PID
        fi
        
        echo "✅ Server stopped successfully"
    else
        echo "⚠️  Server was not running (PID $SERVER_PID not found)"
    fi
    
    # Clean up PID file
    rm -f server.pid
else
    echo "⚠️  No PID file found. Checking for any running FAAAAACES processes..."
    
    # Try to find and kill any running python processes with our script
    PIDS=$(pgrep -f "run_simple.py")
    if [ ! -z "$PIDS" ]; then
        echo "🔄 Found running processes: $PIDS"
        kill $PIDS
        sleep 2
        echo "✅ Processes stopped"
    else
        echo "ℹ️  No running FAAAAACES processes found"
    fi
fi

# Clean up log file if desired
if [ -f "server.log" ]; then
    echo "📄 Server log available at: server.log"
    echo "🗑️  To clean up logs: rm server.log"
fi

echo "🎭 FAAAAACES server shutdown complete"