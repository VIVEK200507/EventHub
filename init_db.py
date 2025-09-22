#!/usr/bin/env python3
"""
Database initialization script for EventHub
Creates sample data for testing and demonstration
"""

from app import app, db, User, Event, EventRegistration
from werkzeug.security import generate_password_hash
from datetime import datetime, date, time, timedelta  # pyright: ignore[reportUnusedImport]
from typing import Any  # type: ignore

def create_sample_data():
    """Create sample users and events for testing"""
    
    # Create sample users
    users = [
        User(  # type: ignore
            username='admin',
            email='admin@eventhub.com',
            password_hash=generate_password_hash('admin123'),
            first_name='Admin',
            last_name='User',
            is_organizer=True
        ),
        User(  # type: ignore
            username='organizer1',
            email='organizer1@eventhub.com',
            password_hash=generate_password_hash('organizer123'),
            first_name='John',
            last_name='Smith',
            is_organizer=True
        ),
        User(  # type: ignore
            username='user1',
            email='user1@eventhub.com',
            password_hash=generate_password_hash('user123'),
            first_name='Jane',
            last_name='Doe',
            is_organizer=False
        ),
        User(  # type: ignore
            username='user2',
            email='user2@eventhub.com',
            password_hash=generate_password_hash('user123'),
            first_name='Bob',
            last_name='Johnson',
            is_organizer=False
        )
    ]
    
    for user in users:
        db.session.add(user)
    
    db.session.commit()
    print("Sample users created successfully!")
    
    # Create sample events
    events = [
        Event(  # type: ignore
            title='Python Web Development Workshop',
            description='Learn how to build web applications using Flask and modern Python techniques. This hands-on workshop covers everything from basic routing to database integration.',
            date=date.today() + timedelta(days=7),
            time=time(10, 0),
            location='Tech Hub, Downtown',
            max_attendees=30,
            price=50.00,
            category='Technology',
            organizer_id=1
        ),
        Event(  # type: ignore
            title='Business Networking Mixer',
            description='Connect with local entrepreneurs and business professionals. Great opportunity to expand your network and discover new business opportunities.',
            date=date.today() + timedelta(days=14),
            time=time(18, 30),
            location='Grand Hotel Ballroom',
            max_attendees=100,
            price=25.00,
            category='Networking',
            organizer_id=2
        ),
        Event(  # type: ignore
            title='Healthy Cooking Class',
            description='Learn to prepare nutritious and delicious meals with our certified nutritionist. All ingredients and equipment provided.',
            date=date.today() + timedelta(days=21),
            time=time(14, 0),
            location='Community Kitchen',
            max_attendees=15,
            price=35.00,
            category='Health & Wellness',
            organizer_id=2
        ),
        Event(  # type: ignore
            title='Art Gallery Opening',
            description='Join us for the opening of our latest contemporary art exhibition featuring works from local and international artists.',
            date=date.today() + timedelta(days=3),
            time=time(19, 0),
            location='Modern Art Gallery',
            max_attendees=80,
            price=0.00,
            category='Arts & Culture',
            organizer_id=1
        ),
        Event(  # type: ignore
            title='Startup Pitch Competition',
            description='Watch innovative startups pitch their ideas to a panel of investors. Great learning opportunity for entrepreneurs.',
            date=date.today() + timedelta(days=28),
            time=time(9, 0),
            location='Innovation Center',
            max_attendees=150,
            price=75.00,
            category='Business',
            organizer_id=1
        ),
        Event(  # type: ignore
            title='Yoga in the Park',
            description='Join us for a relaxing yoga session in the beautiful city park. All levels welcome. Bring your own mat.',
            date=date.today() + timedelta(days=5),
            time=time(8, 0),
            location='Central Park',
            max_attendees=25,
            price=0.00,
            category='Health & Wellness',
            organizer_id=2
        )
    ]
    
    for event in events:
        db.session.add(event)
    
    db.session.commit()
    print("Sample events created successfully!")
    
    # Create sample registrations
    registrations = [
        EventRegistration(user_id=3, event_id=1, status='confirmed'),  # type: ignore
        EventRegistration(user_id=3, event_id=4, status='confirmed'),  # type: ignore
        EventRegistration(user_id=3, event_id=6, status='confirmed'),  # type: ignore
        EventRegistration(user_id=4, event_id=1, status='confirmed'),  # type: ignore
        EventRegistration(user_id=4, event_id=2, status='confirmed'),  # type: ignore
        EventRegistration(user_id=4, event_id=3, status='confirmed'),  # type: ignore
    ]
    
    for registration in registrations:
        db.session.add(registration)
    
    db.session.commit()
    print("Sample registrations created successfully!")

def main():
    """Main function to initialize the database"""
    with app.app_context():
        # Create all tables
        db.create_all()
        print("Database tables created successfully!")
        
        # Check if data already exists
        if User.query.first() is None:
            create_sample_data()
            print("\nSample data created successfully!")
            print("\nSample accounts created:")
            print("Admin: username='admin', password='admin123'")
            print("Organizer: username='organizer1', password='organizer123'")
            print("User: username='user1', password='user123'")
            print("User: username='user2', password='user123'")
        else:
            print("Database already contains data. Skipping sample data creation.")

if __name__ == '__main__':
    main()


