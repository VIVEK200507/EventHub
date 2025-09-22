#!/usr/bin/env python3
"""
Database migration script to add role column to User model
Run this script to update the database schema
"""

from app import app, db, User
from sqlalchemy import text

def migrate_add_role():
    """Add role column to User table"""
    
    with app.app_context():
        try:
            # Check if role column already exists
            with db.engine.connect() as conn:
                result = conn.execute(text("PRAGMA table_info(user)"))
                columns = [row[1] for row in result]
                
                if 'role' in columns:
                    print("Role column already exists in User table.")
                    return
                
                # Add role column with default value 'user'
                conn.execute(text("ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT 'user'"))
                
                # Update existing users based on is_organizer flag
                conn.execute(text("""
                    UPDATE user 
                    SET role = CASE 
                        WHEN is_organizer = 1 THEN 'organizer'
                        ELSE 'user'
                    END
                """))
                
                conn.commit()
            
            # Set admin role for the first user (typically the admin)
            admin_user = User.query.first()
            if admin_user:
                admin_user.role = 'admin'
                db.session.commit()
                print(f"Set admin role for user: {admin_user.username}")
            
            print("✅ Successfully added role column to User table!")
            print("📊 Role distribution:")
            
            # Show role distribution
            with db.engine.connect() as conn:
                roles = conn.execute(text("SELECT role, COUNT(*) FROM user GROUP BY role")).fetchall()
                for role, count in roles:
                    print(f"   - {role}: {count} users")
                
        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            db.session.rollback()
            raise

if __name__ == '__main__':
    migrate_add_role()