#!/usr/bin/env python3
"""
Script to run the database migration for adding role column
"""

from migrate_add_role import migrate_add_role

if __name__ == '__main__':
    print("🔄 Starting database migration...")
    print("📝 Adding role column to User table...")
    
    try:
        migrate_add_role()
        print("\n✅ Migration completed successfully!")
        print("\n📋 Next steps:")
        print("1. Restart your Flask application")
        print("2. Login as admin to access the new admin dashboard")
        print("3. Use the admin dashboard to manage user roles")
        print("\n🎯 Available roles:")
        print("   - admin: Full system access")
        print("   - organizer: Create and manage events")
        print("   - checker: Scan QR codes for events")
        print("   - user: Register for events")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        print("Please check the error and try again.")
