"""
Database migration script to add registration approval fields
Run this script to add the new approval columns to the EventRegistration table
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import config
import os

app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

db = SQLAlchemy(app)

def add_registration_approval_columns():
    """Add approval columns to EventRegistration table"""
    try:
        with app.app_context():
            # Add new columns to EventRegistration table
            with db.engine.connect() as conn:
                try:
                    conn.execute(db.text('''
                        ALTER TABLE event_registration 
                        ADD COLUMN is_approved BOOLEAN DEFAULT FALSE;
                    '''))
                    print("✅ Added is_approved column to EventRegistration")
                except Exception as e:
                    print(f"ℹ️ is_approved column: {e}")
                
                try:
                    conn.execute(db.text('''
                        ALTER TABLE event_registration 
                        ADD COLUMN approved_at DATETIME NULL;
                    '''))
                    print("✅ Added approved_at column to EventRegistration")
                except Exception as e:
                    print(f"ℹ️ approved_at column: {e}")
                
                try:
                    conn.execute(db.text('''
                        ALTER TABLE event_registration 
                        ADD COLUMN approved_by INTEGER NULL;
                    '''))
                    print("✅ Added approved_by column to EventRegistration")
                except Exception as e:
                    print(f"ℹ️ approved_by column: {e}")
                
                conn.commit()
            
            print("✅ EventRegistration approval columns migration completed!")
            print("   Note: Foreign key constraint will be enforced by the ORM model.")
            
    except Exception as e:
        print(f"❌ Error during migration: {e}")

if __name__ == '__main__':
    print("🔄 Adding EventRegistration approval columns to database...")
    add_registration_approval_columns()
    print("✨ Migration completed!")