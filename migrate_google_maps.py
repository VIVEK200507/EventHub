#!/usr/bin/env python3
"""
Database migration to add Google Maps URL field to Event model
"""

from app import app, db
from sqlalchemy import text

def add_google_maps_url_column():
    """Add google_maps_url column to Event table"""
    with app.app_context():
        try:
            # Check if column already exists
            result = db.session.execute(text("""
                SELECT COUNT(*) as count 
                FROM pragma_table_info('event') 
                WHERE name = 'google_maps_url'
            """)).fetchone()
            
            if result and result[0] == 0:
                # Add the google_maps_url column
                db.session.execute(text("""
                    ALTER TABLE event 
                    ADD COLUMN google_maps_url TEXT
                """))
                
                db.session.commit()
                print("✅ Successfully added google_maps_url column to Event table")
            else:
                print("ℹ️  google_maps_url column already exists in Event table")
                
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error adding google_maps_url column: {e}")
            raise

if __name__ == '__main__':
    add_google_maps_url_column()