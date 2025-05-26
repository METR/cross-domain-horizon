#!/usr/bin/env python3

import os
import sys

# Add the parent directory to the path so we can import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

if __name__ == '__main__':
    print("Starting AI Benchmark Baselining Website...")
    print("Open your browser to: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001) 