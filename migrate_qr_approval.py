"""
Database migration script to add QR code approval fields
Run this script to add the new approval columns to the QRCode table
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

def add_qr_approval_columns():
    """Add approval columns to QRCode table"""
    try:
        with app.app_context():
            # Add new columns to QRCode table
            with db.engine.connect() as conn:
                try:
                    conn.execute(db.text('''
                        ALTER TABLE qr_code 
                        ADD COLUMN is_approved BOOLEAN DEFAULT FALSE;
                    '''))
                    print("✅ Added is_approved column")
                except Exception as e:
                    print(f"ℹ️ is_approved column: {e}")
                
                try:
                    conn.execute(db.text('''
                        ALTER TABLE qr_code 
                        ADD COLUMN approved_at DATETIME NULL;
                    '''))
                    print("✅ Added approved_at column")
                except Exception as e:
                    print(f"ℹ️ approved_at column: {e}")
                
                try:
                    conn.execute(db.text('''
                        ALTER TABLE qr_code 
                        ADD COLUMN approved_by INTEGER NULL;
                    '''))
                    print("✅ Added approved_by column")
                except Exception as e:
                    print(f"ℹ️ approved_by column: {e}")
                
                conn.commit()
            
            print("✅ QR code approval columns migration completed!")
            print("   Note: Foreign key constraint will be enforced by the ORM model.")
            
    except Exception as e:
        print(f"❌ Error during migration: {e}")

if __name__ == '__main__':
    print("🔄 Adding QR code approval columns to database...")
    add_qr_approval_columns()
    print("✨ Migration completed!")