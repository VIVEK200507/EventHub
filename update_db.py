#!/usr/bin/env python3
"""
Database update script to add new tables
"""

from app import app, db

def update_database():
    """Add new tables to existing database"""
    with app.app_context():
        # Create all tables (this will add new tables)
        db.create_all()
        print("Database updated successfully!")
        print("New tables have been added:")
        print("- QRCode table")
        print("- Attendance table") 
        print("- EventFeedback table")
        print("- FaceEmbedding table")

if __name__ == '__main__':
    update_database()
