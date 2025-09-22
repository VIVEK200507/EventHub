# Role-Based Authorization System

This document describes the new role-based authorization system implemented in the Event Management System.

## Overview

The system now supports four distinct user roles with different permissions:

- **Admin**: Full system access, can manage users and all events
- **Organizer**: Can create and manage events, scan QR codes for their events
- **Checker**: Can scan QR codes for any event (security guards, volunteers, students)
- **User**: Can register for events (default role)

## Database Changes

### New User Model Column
```python
role = db.Column(db.String(20), default='user')  # Possible roles: admin, organizer, checker, user
```

### Migration
Run the migration script to add the role column to existing users:
```bash
python run_migration.py
```

## Role Permissions

### Admin (`admin`)
- ✅ Full system access
- ✅ Manage all users (assign roles, delete users)
- ✅ View all events and registrations
- ✅ Scan QR codes for any event
- ✅ Create and manage events

### Organizer (`organizer`)
- ✅ Create and manage their own events
- ✅ View registrations for their events
- ✅ Scan QR codes for their events
- ❌ Cannot manage other users
- ❌ Cannot access other organizers' events

### Checker (`checker`)
- ✅ Scan QR codes for any event
- ✅ View event details for scanning
- ❌ Cannot create or manage events
- ❌ Cannot manage users
- ❌ Cannot view event registrations

### User (`user`)
- ✅ Register for events
- ✅ View their own registrations
- ❌ Cannot create events
- ❌ Cannot scan QR codes
- ❌ Cannot manage users

## New Features

### 1. Admin Dashboard
- **Route**: `/dashboard` (for admin users)
- **Template**: `templates/admin_dashboard.html`
- **Features**:
  - System overview with user and event statistics
  - Recent users list
  - Recent events list
  - Quick access to user management

### 2. User Management
- **Route**: `/admin/users`
- **Template**: `templates/admin_users.html`
- **Features**:
  - List all users with pagination
  - Search users by name, username, or email
  - Filter users by role
  - Update user roles
  - Delete users (with confirmation)

### 3. Role-Based Navigation
- Navigation menu shows different options based on user role
- User avatar shows role-specific icon
- Role badge displayed next to username
- Admin users see "Manage Users" option

### 4. Enhanced QR Scanner Access
- **Organizers**: Can scan QR codes for their own events
- **Admins**: Can scan QR codes for any event
- **Checkers**: Can scan QR codes for any event
- **Users**: Cannot access QR scanner

## API Changes

### User Model Methods
```python
def is_admin(self):
    return self.role == 'admin'

def is_organizer_role(self):
    return self.role == 'organizer'

def is_checker(self):
    return self.role == 'checker'

def can_scan_qr(self):
    return self.role in ['admin', 'organizer', 'checker']

def can_manage_events(self):
    return self.role in ['admin', 'organizer']

def can_manage_users(self):
    return self.role == 'admin'
```

### Updated Routes
- `/dashboard` - Now shows admin dashboard for admin users
- `/admin/users` - New admin user management page
- `/admin/users/<id>/update_role` - Update user role
- `/admin/users/<id>/delete` - Delete user
- `/scanner` - Updated to allow checkers and admins
- `/scanner/event/<id>` - Updated role-based access

## Event Registrations Enhancement

The event registrations page (`templates/event_registrations.html`) already shows:
- ✅ User's email (`registration.user.email`)
- ✅ Registration date (`registration.registration_date`)

## QR Code Scanning Enhancement

The QR scanning system now:
- ✅ Records who scanned the QR code (`scanned_by = current_user.id`)
- ✅ Allows both organizers and checkers to scan
- ✅ Shows appropriate events based on user role

## Installation & Setup

1. **Run the migration**:
   ```bash
   python run_migration.py
   ```

2. **Restart your Flask application**

3. **Login as admin** to access the new admin dashboard

4. **Assign roles** to users through the admin interface

## Security Considerations

- Admins cannot remove their own admin privileges
- Admins cannot delete their own account
- Role changes are logged and require confirmation
- All role-based access is checked server-side

## Backward Compatibility

- Existing `is_organizer` field is maintained for backward compatibility
- The system automatically sets `is_organizer=True` when role is set to 'organizer'
- All existing functionality continues to work

## Example Usage

### Assigning a Checker Role
1. Login as admin
2. Go to "Manage Users"
3. Find the user you want to make a checker
4. Click "Role" dropdown
5. Select "Set as Checker"
6. Confirm the change

### Creating a Checker Account
1. User registers normally
2. Admin assigns "checker" role
3. User can now access QR scanner for any event

## Troubleshooting

### Migration Issues
- Ensure database is not locked
- Check that all users have valid data
- Verify SQLite permissions

### Role Assignment Issues
- Check that user exists
- Verify admin permissions
- Ensure role value is valid

### Access Issues
- Clear browser cache
- Check user role in database
- Verify route permissions
