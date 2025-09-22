#!/usr/bin/env python3
"""
Run script for EventHub - Event Management System
"""

import os
from app import app, db

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")
    
    # Get configuration from environment
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print(f"Starting EventHub server...")
    print(f"Debug mode: {debug}")
    print(f"Server running at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=debug, host=host, port=port)




