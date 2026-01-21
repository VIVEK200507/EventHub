from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
from typing import Optional, Any
import os
import qrcode
import io
import base64
import cv2
import numpy as np
import pickle
import re
from werkzeug.utils import secure_filename
from config import config
import json

app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Face recognition configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
FACE_MATCH_THRESHOLD = 0.85  # 85% similarity threshold

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # type: ignore 
login_manager.login_message = 'Please log in to access this page.'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    is_organizer = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # New fields
    role = db.Column(db.String(50), nullable=False)  # student, faculty, technician
    details = db.Column(db.String(100), nullable=True)  # course/department/type

    # Relationships
    organized_events = db.relationship('Event', backref='organizer', lazy=True)
    registrations = db.relationship('EventRegistration', foreign_keys='EventRegistration.user_id', backref='user', lazy=True)
    
    # Role-based helper methods
    def is_admin(self):
        return self.role == 'admin'
    
    def is_organizer_role(self):
        return self.role == 'organizer'
    
    def is_checker(self):
        return self.role == 'checker'
    
    def can_scan_qr(self):
        return self.role in ['admin', 'organizer', 'checker']or self.is_organizer
    
    def can_manage_events(self):
        return self.role in ['admin', 'organizer']
    
    def can_manage_users(self):
        return self.role == 'admin'

# Event Model
class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    location = db.Column(db.String(200), nullable=False)
    google_maps_url = db.Column(db.Text, nullable=True)  # Google Maps location URL
    max_attendees = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, default=0.0)
    category = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='active')  # active, cancelled, completed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organizer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    registrations = db.relationship('EventRegistration', backref='event', lazy=True, cascade='all, delete-orphan')

# Event Registration Model
class EventRegistration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    event_id = db.Column(db.Integer, db.ForeignKey('event.id'), nullable=False)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='confirmed')  # confirmed, cancelled, waitlist
    
    # Host approval fields
    is_approved = db.Column(db.Boolean, default=False)  # Track if registration is approved by host
    approved_at = db.Column(db.DateTime, nullable=True)  # When registration was approved
    approved_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Host who approved
    
    # Unique constraint to prevent duplicate registrations
    __table_args__ = (db.UniqueConstraint('user_id', 'event_id', name='unique_user_event_registration'),)
    
    # Relationship with QR code
    qr_code = db.relationship('QRCode', backref='registration', uselist=False, cascade='all, delete-orphan')
    approver = db.relationship('User', foreign_keys=[approved_by], backref='approved_registrations', post_update=True)

# QR Code Model
class QRCode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    registration_id = db.Column(db.Integer, db.ForeignKey('event_registration.id'), nullable=False)
    qr_data = db.Column(db.Text, nullable=False)  # Base64 encoded QR code image
    qr_text = db.Column(db.Text, nullable=False)  # Text data encoded in QR code
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_used = db.Column(db.Boolean, default=False)  # Track if QR code has been scanned
    used_at = db.Column(db.DateTime, nullable=True)
    
    # Pass approval fields
    is_approved = db.Column(db.Boolean, default=False)  # Track if pass has been approved by host
    approved_at = db.Column(db.DateTime, nullable=True)  # When pass was approved
    approved_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Host who approved
    
    # Relationship with attendance
    attendance = db.relationship('Attendance', backref='qr_code', uselist=False, cascade='all, delete-orphan')
    approver = db.relationship('User', foreign_keys=[approved_by], backref='approved_qr_codes', post_update=True)

# Attendance Model
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    qr_code_id = db.Column(db.Integer, db.ForeignKey('qr_code.id'), nullable=False)
    registration_id = db.Column(db.Integer, db.ForeignKey('event_registration.id'), nullable=False)
    event_id = db.Column(db.Integer, db.ForeignKey('event.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scanned_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Organizer who scanned
    scan_time = db.Column(db.DateTime, default=datetime.utcnow)
    scan_location = db.Column(db.String(200), nullable=True)  # Optional location tracking
    notes = db.Column(db.Text, nullable=True)  # Optional notes from scanner
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id], backref='attendance_records')
    scanner = db.relationship('User', foreign_keys=[scanned_by], backref='scanned_attendance')
    event = db.relationship('Event', backref='attendance_records')
    registration = db.relationship('EventRegistration', backref='attendance_record')

# Event Feedback Model
class EventFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey('event.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    registration_id = db.Column(db.Integer, db.ForeignKey('event_registration.id'), nullable=True)
    
    # Rating fields (1-5 scale)
    overall_rating = db.Column(db.Integer, nullable=False)  # Overall event rating
    content_rating = db.Column(db.Integer, nullable=True)   # Content quality
    organization_rating = db.Column(db.Integer, nullable=True)  # Event organization
    venue_rating = db.Column(db.Integer, nullable=True)     # Venue quality
    value_rating = db.Column(db.Integer, nullable=True)     # Value for money
    
    # Text feedback
    positive_feedback = db.Column(db.Text, nullable=True)   # What went well
    improvement_suggestions = db.Column(db.Text, nullable=True)  # Areas for improvement
    additional_comments = db.Column(db.Text, nullable=True)  # Additional comments
    
    # Recommendation
    would_recommend = db.Column(db.Boolean, nullable=True)  # Would recommend to others
    
    # Metadata
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_anonymous = db.Column(db.Boolean, default=False)  # Anonymous feedback option
    
    # Relationships
    event = db.relationship('Event', backref='feedback_records')
    user = db.relationship('User', backref='feedback_submissions')
    registration = db.relationship('EventRegistration', backref='feedback')
    
    # Unique constraint to prevent duplicate feedback
    __table_args__ = (db.UniqueConstraint('event_id', 'user_id', name='unique_user_event_feedback'),)

# Face Embedding Model
class FaceEmbedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    registration_id = db.Column(db.Integer, db.ForeignKey('event_registration.id'), nullable=True)
    
    # Face data
    face_embedding = db.Column(db.LargeBinary, nullable=False)  # Pickled face encoding
    photo_path = db.Column(db.String(255), nullable=False)  # Path to uploaded photo
    face_detected = db.Column(db.Boolean, default=True)  # Whether face was detected in photo
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_verified = db.Column(db.Boolean, default=False)  # Whether face has been verified at event
    
    # Relationships
    user = db.relationship('User', backref='face_embeddings')
    registration = db.relationship('EventRegistration', backref='face_embedding')
    
    # Unique constraint to prevent duplicate face embeddings per user
    __table_args__ = (db.UniqueConstraint('user_id', name='unique_user_face_embedding'),)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# QR Code Generation Function
def generate_qr_code(registration):
    """Generate QR code for event registration"""
    # Create QR code data
    qr_data = {
        'registration_id': registration.id,
        'user_id': registration.user_id,
        'event_id': registration.event_id,
        'user_name': f"{registration.user.first_name} {registration.user.last_name}",
        'event_title': registration.event.title,
        'event_date': registration.event.date.isoformat(),
        'event_time': registration.event.time.isoformat(),
        'registration_date': registration.registration_date.isoformat(),
        'verification_code': f"EVENT{registration.id:06d}{registration.user_id:04d}"
    }
    
    # Convert to JSON string
    import json
    qr_text = json.dumps(qr_data, indent=2)
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # type: ignore  # type: ignore
        box_size=10,
        border=4,
    )
    qr.add_data(qr_text)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 string
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')  # type: ignore  # type: ignore
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return qr_base64, qr_text

# Face Recognition Utility Functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_google_maps_url(url):
    """Validate if the URL is a proper Google Maps URL"""
    if not url or not url.strip():
        return True  # Empty URL is valid (optional field)
    
    # Google Maps URL patterns
    google_maps_patterns = [
        r'^https://www\.google\.[a-z]{2,3}/maps',
        r'^https://maps\.google\.[a-z]{2,3}/',
        r'^https://goo\.gl/maps/',
        r'^https://maps\.app\.goo\.gl/',
        r'^https://www\.google\.[a-z]{2,3}/maps/place/',
        r'^https://maps\.google\.[a-z]{2,3}/maps\?',
    ]
    
    url = url.strip()
    
    # Check if URL matches any Google Maps pattern
    for pattern in google_maps_patterns:
        if re.match(pattern, url, re.IGNORECASE):
            return True
    
    return False

def extract_google_maps_embed_url(url):
    """Extract embeddable URL from Google Maps URL"""
    if not url or not validate_google_maps_url(url):
        return None
    
    # If it's already an embed URL, return as is
    if 'embed' in url:
        return url
    
    # Try to extract place information for embed
    # This is a simplified version - in production you might want more sophisticated parsing
    if 'place/' in url:
        # Extract place ID or name from the URL
        place_match = re.search(r'/place/([^/]+)', url)
        if place_match:
            place_info = place_match.group(1)
            return f"https://www.google.com/maps/embed/v1/place?key=YOUR_API_KEY&q={place_info}"
    
    # For other URLs, create a basic embed URL
    # Note: In production, you should use a proper Google Maps API key
    return url  # Return original URL for now

def detect_and_encode_face(image_path):
    """Detect face in image and return face features"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, False
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore  # type: ignore
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, False  # No face detected
        
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to standard size for comparison
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Flatten and normalize
        face_features = face_roi.flatten().astype(np.float32) / 255.0
        
        return face_features, True
        
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None, False

def compare_faces(known_features, unknown_features):
    """Compare two face feature vectors and return similarity score"""
    try:
        # Calculate cosine similarity
        dot_product = np.dot(known_features, unknown_features)
        norm_known = np.linalg.norm(known_features)
        norm_unknown = np.linalg.norm(unknown_features)
        
        if norm_known == 0 or norm_unknown == 0:
            return 0.0
        
        cosine_similarity = dot_product / (norm_known * norm_unknown)
        
        # Convert to percentage (0-100%)
        similarity = max(0, cosine_similarity * 100)
        
        return similarity
        
    except Exception as e:
        print(f"Error in face comparison: {e}")
        return 0.0

def save_face_embedding(user_id, photo_path, registration_id=None):
    """Save face embedding to database"""
    try:
        # Detect and encode face
        face_features, face_detected = detect_and_encode_face(photo_path)
        
        if not face_detected or face_features is None:
            return False, "No face detected in the uploaded photo"
        
        # Pickle the face features for storage
        face_embedding_data = pickle.dumps(face_features)
        
        # Check if user already has a face embedding
        existing_embedding = FaceEmbedding.query.filter_by(user_id=user_id).first()
        
        if existing_embedding:
            # Update existing embedding
            existing_embedding.face_embedding = face_embedding_data
            existing_embedding.photo_path = photo_path
            existing_embedding.face_detected = face_detected
            existing_embedding.registration_id = registration_id
            existing_embedding.is_verified = False
        else:
            # Create new embedding
            face_embedding = FaceEmbedding(  # type: ignore
                user_id=user_id,
                registration_id=registration_id,
                face_embedding=face_embedding_data,
                photo_path=photo_path,
                face_detected=face_detected
            )
            db.session.add(face_embedding)
        
        db.session.commit()
        return True, "Face embedding saved successfully"
        
    except Exception as e:
        db.session.rollback()
        print(f"Error saving face embedding: {e}")
        return False, f"Error saving face embedding: {str(e)}"

# Routes
@app.route('/')
def index():
    events = Event.query.filter_by(status='active').order_by(Event.date.asc()).limit(6).all()
    return render_template('index.html', events=events)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        role = request.form['role']
        details = request.form.get('details', '')  # course/department/type
        is_organizer = 'is_organizer' in request.form

        # Validation
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'error')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'error')
            return render_template('register.html')

        # Create user
        user = User(  # type: ignore
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            first_name=first_name,
            last_name=last_name,
            role=role,
            details=details,
            is_organizer=is_organizer
        )
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_admin():
        # Admin dashboard - show all users and system overview
        users = User.query.order_by(User.created_at.desc()).all()
        events = Event.query.order_by(Event.created_at.desc()).limit(10).all()
        return render_template('admin_dashboard.html', users=users, events=events)
    elif current_user.is_organizer_role() or current_user.is_organizer:
        # Show events organized by user
        organized_events = Event.query.filter_by(organizer_id=current_user.id).order_by(Event.date.desc()).all()
        return render_template('organizer_dashboard.html', events=organized_events)
    else:
        # Show events user is registered for
        registrations = EventRegistration.query.filter_by(user_id=current_user.id).all()
        today_date = date.today()
        return render_template('user_dashboard.html', registrations=registrations, today_date=today_date)

@app.route('/events')
def events():
    page = request.args.get('page', 1, type=int)
    category = request.args.get('category', '')
    search = request.args.get('search', '')
    
    query = Event.query.filter_by(status='active')
    
    if category:
        query = query.filter_by(category=category)
    
    if search:
        query = query.filter(Event.title.contains(search) | Event.description.contains(search))
    
    events = query.order_by(Event.date.asc()).paginate(
        page=page, per_page=9, error_out=False
    )
    
    categories = db.session.query(Event.category.distinct()).all()
    categories = [cat[0] for cat in categories]
    
    return render_template('events.html', events=events, categories=categories)

@app.route('/event/<int:event_id>')
def event_detail(event_id):
    event = Event.query.get_or_404(event_id)
    is_registered = False
    registration = None
    
    if current_user.is_authenticated:
        registration = EventRegistration.query.filter_by(
            user_id=current_user.id, 
            event_id=event_id
        ).first()
        is_registered = registration is not None
    
    attendees_count = EventRegistration.query.filter_by(event_id=event_id, status='confirmed').count()
    
    return render_template('event_detail.html', 
                         event=event, 
                         is_registered=is_registered,
                         registration=registration,
                         attendees_count=attendees_count)

@app.route('/create_event', methods=['GET', 'POST'])
@login_required
def create_event():
    if not current_user.is_organizer:
        flash('Only organizers can create events!', 'error')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Get form data
        google_maps_url = request.form.get('google_maps_url', '').strip()
        
        # Validate Google Maps URL if provided
        if google_maps_url and not validate_google_maps_url(google_maps_url):
            flash('Please enter a valid Google Maps URL. URLs should start with https://maps.google.com or https://www.google.com/maps', 'error')
            return render_template('create_event.html')
        
        event = Event(  # type: ignore
            title=request.form['title'],
            description=request.form['description'],
            date=datetime.strptime(request.form['date'], '%Y-%m-%d').date(),
            time=datetime.strptime(request.form['time'], '%H:%M').time(),
            location=request.form['location'],
            google_maps_url=google_maps_url if google_maps_url else None,
            max_attendees=int(request.form['max_attendees']),
            price=float(request.form['price']),
            category=request.form['category'],
            organizer_id=current_user.id
        )
        
        db.session.add(event)
        db.session.commit()
        
        flash('Event created successfully!', 'success')
        return redirect(url_for('event_detail', event_id=event.id))
    
    return render_template('create_event.html')

@app.route('/register_event/<int:event_id>')
@login_required
def register_event(event_id):
    event = Event.query.get_or_404(event_id)
    
    # Check if already registered
    existing_registration = EventRegistration.query.filter_by(
        user_id=current_user.id, 
        event_id=event_id
    ).first()
    
    if existing_registration:
        flash('You are already registered for this event!', 'error')
        return redirect(url_for('event_detail', event_id=event_id))
    
    # Check if event is full
    current_attendees = EventRegistration.query.filter_by(event_id=event_id).count()
    if current_attendees >= event.max_attendees:
        flash('This event is full!', 'error')
        return redirect(url_for('event_detail', event_id=event_id))

    
    # Register user (without generating QR code yet)
    registration = EventRegistration(  # type: ignore
        user_id=current_user.id,
        event_id=event_id,
        is_approved=False  # Registration needs host approval
    )
    
    db.session.add(registration)
    db.session.commit()
    
    # Show welcome message and approval notice
    flash(f'Welcome to {event.title}! Your registration is received. Your QR pass will be visible after the event host approves your registration.', 'success')
    
    return redirect(url_for('event_detail', event_id=event_id))

@app.route('/event/<int:event_id>/approve_registration/<int:registration_id>')
@login_required
def approve_single_registration(event_id, registration_id):
    """Approve a single participant registration and generate QR code"""
    if not current_user.is_organizer:
        flash('Only organizers can approve registrations!', 'error')
        return redirect(url_for('dashboard'))
    
    event = Event.query.get_or_404(event_id)
    
    if event.organizer_id != current_user.id:
        flash('You can only approve registrations for your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    registration = EventRegistration.query.get_or_404(registration_id)
    
    if registration.event_id != event_id:
        flash('Registration does not belong to this event!', 'error')
        return redirect(url_for('event_registrations', event_id=event_id))
    
    if registration.is_approved:
        flash('Registration is already approved!', 'info')
        return redirect(url_for('event_registrations', event_id=event_id))
    
    # Approve the registration
    registration.is_approved = True
    registration.approved_at = datetime.utcnow()
    registration.approved_by = current_user.id
    
    # Generate QR code now that registration is approved
    try:
        qr_base64, qr_text = generate_qr_code(registration)
        
        # Store QR code in database with approval
        qr_code = QRCode(  # type: ignore
            registration_id=registration.id,
            qr_data=qr_base64,
            qr_text=qr_text,
            is_approved=True,
            approved_at=datetime.utcnow(),
            approved_by=current_user.id
        )
        
        db.session.add(qr_code)
        db.session.commit()
        
        flash(f'Successfully approved registration for {registration.user.first_name} {registration.user.last_name} and generated QR pass!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Registration approved but QR code generation failed: {str(e)}', 'warning')
    
    return redirect(url_for('event_registrations', event_id=event_id))

@app.route('/event/<int:event_id>/approve_all')
@login_required
def issue_event_passes(event_id):
    """Approve all pending registrations and generate QR codes for an event (organizers only)"""
    if not current_user.is_organizer:
        flash('Only organizers can approve registrations!', 'error')
        return redirect(url_for('dashboard'))
    
    event = Event.query.get_or_404(event_id)
    
    if event.organizer_id != current_user.id:
        flash('You can only approve registrations for your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    # Find all pending registrations for this event
    pending_registrations = EventRegistration.query.filter_by(
        event_id=event_id,
        is_approved=False
    ).all()
    
    if not pending_registrations:
        flash('No pending registrations to approve for this event.', 'info')
        return redirect(url_for('event_registrations', event_id=event_id))
    
    # Approve all pending registrations and generate QR codes
    approved_count = 0
    failed_count = 0
    
    for registration in pending_registrations:
        try:
            # Approve the registration
            registration.is_approved = True
            registration.approved_at = datetime.utcnow()
            registration.approved_by = current_user.id
            
            # Generate QR code
            qr_base64, qr_text = generate_qr_code(registration)
            
            # Store QR code in database
            qr_code = QRCode(  # type: ignore
                registration_id=registration.id,
                qr_data=qr_base64,
                qr_text=qr_text,
                is_approved=True,
                approved_at=datetime.utcnow(),
                approved_by=current_user.id
            )
            
            db.session.add(qr_code)
            approved_count += 1
            
        except Exception as e:
            print(f"Error approving registration {registration.id}: {e}")
            failed_count += 1
    
    db.session.commit()
    
    if approved_count > 0:
        flash(f'Successfully approved {approved_count} registrations and generated QR passes for "{event.title}"! Participants can now view and download their passes.', 'success')
    
    if failed_count > 0:
        flash(f'Warning: {failed_count} registrations could not be processed. Please try again or contact support.', 'warning')
    
    return redirect(url_for('event_registrations', event_id=event_id))

@app.route('/cancel_registration/<int:event_id>')
@login_required
def cancel_registration(event_id):
    registration = EventRegistration.query.filter_by(
        user_id=current_user.id, 
        event_id=event_id
    ).first()
    
    if registration:
        db.session.delete(registration)
        db.session.commit()
        flash('Registration cancelled successfully!', 'success')
    else:
        flash('Registration not found!', 'error')
    
    return redirect(url_for('event_detail', event_id=event_id))

@app.route('/edit_event/<int:event_id>', methods=['GET', 'POST'])
@login_required
def edit_event(event_id):
    event = Event.query.get_or_404(event_id)
    
    if event.organizer_id != current_user.id:
        flash('You can only edit your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Get form data
        google_maps_url = request.form.get('google_maps_url', '').strip()
        
        # Validate Google Maps URL if provided
        if google_maps_url and not validate_google_maps_url(google_maps_url):
            flash('Please enter a valid Google Maps URL. URLs should start with https://maps.google.com or https://www.google.com/maps', 'error')
            return render_template('edit_event.html', event=event)
        
        event.title = request.form['title']
        event.description = request.form['description']
        event.date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
        event.time = datetime.strptime(request.form['time'], '%H:%M').time()
        event.location = request.form['location']
        event.google_maps_url = google_maps_url if google_maps_url else None
        event.max_attendees = int(request.form['max_attendees'])
        event.price = float(request.form['price'])
        event.category = request.form['category']
        event.updated_at = datetime.utcnow()
        
        db.session.commit()
        flash('Event updated successfully!', 'success')
        return redirect(url_for('event_detail', event_id=event.id))
    
    return render_template('edit_event.html', event=event)

@app.route('/delete_event/<int:event_id>')
@login_required
def delete_event(event_id):
    event = Event.query.get_or_404(event_id)
    
    if event.organizer_id != current_user.id:
        flash('You can only delete your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    db.session.delete(event)
    db.session.commit()
    flash('Event deleted successfully!', 'success')
    return redirect(url_for('dashboard'))



@app.route('/qr_code/<int:registration_id>')
@login_required
def view_qr_code(registration_id):
    """Display QR code for a registration"""
    registration = EventRegistration.query.get_or_404(registration_id)
    
    # Check if user has permission to view this QR code
    if registration.user_id != current_user.id and not current_user.is_organizer:
        flash('You can only view your own QR codes!', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if registration is approved
    if not registration.is_approved:
        if registration.user_id == current_user.id:
            flash('Your registration is still pending approval by the event host.', 'warning')
            return redirect(url_for('dashboard'))
    
    # Check if QR code exists
    if not registration.qr_code:
        if registration.user_id == current_user.id:
            flash('Your QR pass will be available after host approval.', 'info')
        else:
            flash('QR code not yet generated for this registration.', 'info')
        return redirect(url_for('dashboard'))
    
    qr_code = registration.qr_code

    # ✅ Parse JSON safely
    try:
        qr_data = json.loads(qr_code.qr_text)
    except Exception:
        qr_data = {}  # Always define qr_data so template won't crash
        flash("Could not parse QR code data", "warning")

    return render_template(
        'qr_code.html',
        registration=registration,
        qr_code=qr_code,
        qr_data=qr_data   # ✅ Now template has qr_data
    )

@app.route('/qr_code/<int:registration_id>/download')
@login_required
def download_qr_code(registration_id):
    """Download QR code as PNG file"""
    registration = EventRegistration.query.get_or_404(registration_id)
    
    # Check if user has permission to download this QR code
    if registration.user_id != current_user.id and not current_user.is_organizer:
        flash('You can only download your own QR codes!', 'error')
        return redirect(url_for('dashboard'))
    
    qr_code = QRCode.query.filter_by(registration_id=registration.id).first()
    if not qr_code:
        flash('QR code not found for this registration!', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if QR code is approved (unless organizer)
    if not qr_code.is_approved and not current_user.is_organizer:
        flash('QR pass is not yet approved by the event host. Please wait for approval.', 'warning')
        return redirect(url_for('view_qr_code', registration_id=registration_id))

    
    # Decode base64 image
    qr_image_data = base64.b64decode(qr_code.qr_data)
    
    # Create filename
    filename = f"event_{registration.event.id}_registration_{registration.id}_qr.png"
    
    return send_file(
        io.BytesIO(qr_image_data),
        mimetype='image/png',
        as_attachment=True,
        download_name=filename
    )

@app.route('/qr_code/<int:registration_id>/verify')
@login_required
def verify_qr_code(registration_id):
    """Verify and mark QR code as used (for organizers)"""
    if not current_user.is_organizer:
        flash('Only organizers can verify QR codes!', 'error')
        return redirect(url_for('dashboard'))
    
    registration = EventRegistration.query.get_or_404(registration_id)
    qr_code = QRCode.query.filter_by(registration_id=registration.id).first()
    if not qr_code:
        flash('QR code not found for this registration!', 'error')
        return redirect(url_for('dashboard'))

    
    if qr_code.is_used:
        flash('This QR code has already been used!', 'warning')
    else:
        qr_code.is_used = True
        qr_code.used_at = datetime.utcnow()
        db.session.commit()
        flash('QR code verified successfully!', 'success')
    
    return redirect(url_for('view_qr_code', registration_id=registration_id))

@app.route('/event/<int:event_id>/registrations')
@login_required
def event_registrations(event_id):
    """View all registrations for an event (organizers only)"""
    event = Event.query.get_or_404(event_id)
    
    if event.organizer_id != current_user.id:
        flash('You can only view registrations for your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    registrations = EventRegistration.query.filter_by(event_id=event_id).all()
    
    return render_template('event_registrations.html', event=event, registrations=registrations)

# QR Code Scanner Routes
@app.route('/scanner')
@login_required
def qr_scanner():
    """QR code scanner interface for organizers and checkers"""
    if not current_user.can_scan_qr():
        flash('You are not authorized to scan QR codes!', 'error')
        return redirect(url_for('dashboard'))
    
    # Get events based on user role
    if current_user.is_admin():
        # Admin can scan for all events
        events = Event.query.filter_by(status='active').all()
    elif current_user.is_organizer_role() or current_user.is_organizer:
        # Organizers can scan for their own events
        events = Event.query.filter_by(organizer_id=current_user.id, status='active').all()
    else:
        # Checkers can scan for all active events
        events = Event.query.filter_by(status='active').all()
    
    return render_template('qr_scanner.html', events=events)

@app.route('/scanner/event/<int:event_id>')
@login_required
def event_scanner(event_id):
    """Event-specific QR scanner"""
    if not current_user.can_scan_qr():
        flash('You are not authorized to scan QR codes!', 'error')
        return redirect(url_for('dashboard'))
    
    event = Event.query.get_or_404(event_id)
    
    # Check if current user can scan for this event (organizer, admin, or checker)
    if event.organizer_id != current_user.id and not current_user.is_admin() and not current_user.is_checker():
        flash('You can only scan QR codes for events you organize!', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('event_scanner.html', event=event)

@app.route('/api/scan_qr', methods=['POST'])
@login_required
def scan_qr_code():
    """API endpoint to process scanned QR code data"""
    if not current_user.can_scan_qr():
        return jsonify({'success': False, 'error': 'You are not authorized to scan QR codes!'}), 403
    
    try:
        data = request.get_json()
        qr_data = data.get('qr_data')
        event_id = data.get('event_id')
        scan_location = data.get('scan_location', '')
        notes = data.get('notes', '')
        
        if not qr_data:
            return jsonify({'success': False, 'error': 'No QR code data provided'}), 400
        
        # Parse QR code data
        import json
        try:
            qr_info = json.loads(qr_data)
        except json.JSONDecodeError:
            return jsonify({'success': False, 'error': 'Invalid QR code format'}), 400
        
        # Validate QR code structure
        required_fields = ['registration_id', 'user_id', 'event_id', 'verification_code']
        for field in required_fields:
            if field not in qr_info:
                return jsonify({'success': False, 'error': f'Missing {field} in QR code'}), 400
        
        # Find the registration
        registration = EventRegistration.query.get(qr_info['registration_id'])
        if not registration:
            return jsonify({'success': False, 'error': 'Registration not found'}), 404
        
        # Verify event matches
        if event_id and int(qr_info['event_id']) != event_id:
            return jsonify({'success': False, 'error': 'QR code is for a different event'}), 400
        
        # Check if already scanned
        existing_attendance = Attendance.query.filter_by(registration_id=registration.id).first()
        if existing_attendance:
            return jsonify({
                'success': False, 
                'error': 'Already checked in',
                'attendance_info': {
                    'scan_time': existing_attendance.scan_time.isoformat(),
                    'scanned_by': existing_attendance.scanner.username
                }
            }), 409
        
        # Verify QR code exists and is valid
        qr_code = registration.qr_code
        if not qr_code:
            return jsonify({'success': False, 'error': 'QR code not found for this registration'}), 404
        
        # Create attendance record
        attendance = Attendance(  # type: ignore
            qr_code_id=qr_code.id,
            registration_id=registration.id,
            event_id=registration.event_id,
            user_id=registration.user_id,
            scanned_by=current_user.id,
            scan_location=scan_location,
            notes=notes
        )
        
        # Mark QR code as used
        qr_code.is_used = True
        qr_code.used_at = datetime.utcnow()
        
        db.session.add(attendance)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Check-in successful!',
            'attendance_info': {
                'user_name': f"{registration.user.first_name} {registration.user.last_name}",
                'event_title': registration.event.title,
                'scan_time': attendance.scan_time.isoformat(),
                'verification_code': qr_info['verification_code']
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'Scan failed: {str(e)}'}), 500

@app.route('/event/<int:event_id>/attendance')
@login_required
def event_attendance(event_id):
    """View attendance records for an event"""
    if not current_user.is_organizer:
        flash('Only organizers can view attendance records!', 'error')
        return redirect(url_for('dashboard'))
    
    event = Event.query.get_or_404(event_id)
    
    if event.organizer_id != current_user.id:
        flash('You can only view attendance for your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    # Get attendance records
    attendance_records = Attendance.query.filter_by(event_id=event_id).order_by(Attendance.scan_time.desc()).all()
    
    # Get statistics
    total_registrations = EventRegistration.query.filter_by(event_id=event_id).count()
    total_attendance = len(attendance_records)
    attendance_rate = (total_attendance / total_registrations * 100) if total_registrations > 0 else 0

    
    return render_template('event_attendance.html', 
                         event=event, 
                         attendance_records=attendance_records,
                         total_registrations=total_registrations,
                         total_attendance=total_attendance,
                         attendance_rate=attendance_rate)

# Event Feedback Routes
@app.route('/event/<int:event_id>/feedback')
@login_required
def event_feedback(event_id):
    """Feedback form for an event"""
    event = Event.query.get_or_404(event_id)
    
    # Check if user is registered for the event
    registration = EventRegistration.query.filter_by(
        user_id=current_user.id, 
        event_id=event_id
    ).first()
    
    if not registration:
        flash('You must be registered for this event to provide feedback!', 'error')
        return redirect(url_for('event_detail', event_id=event_id))
    
    # Check if user has already submitted feedback
    existing_feedback = EventFeedback.query.filter_by(
        event_id=event_id,
        user_id=current_user.id
    ).first()
    
    if existing_feedback:
        flash('You have already submitted feedback for this event!', 'info')
        return redirect(url_for('view_feedback', feedback_id=existing_feedback.id))
    
    return render_template('event_feedback.html', event=event, registration=registration)

@app.route('/event/<int:event_id>/feedback/submit', methods=['POST'])
@login_required
def submit_feedback(event_id):
    """Submit feedback for an event"""
    event = Event.query.get_or_404(event_id)
    
    # Check if user is registered for the event
    registration = EventRegistration.query.filter_by(
        user_id=current_user.id, 
        event_id=event_id
    ).first()
    
    if not registration:
        flash('You must be registered for this event to provide feedback!', 'error')
        return redirect(url_for('event_detail', event_id=event_id))
    
    # Check if user has already submitted feedback
    existing_feedback = EventFeedback.query.filter_by(
        event_id=event_id,
        user_id=current_user.id
    ).first()
    
    if existing_feedback:
        flash('You have already submitted feedback for this event!', 'error')
        return redirect(url_for('view_feedback', feedback_id=existing_feedback.id))
    
    # Create feedback record
    feedback = EventFeedback(  # type: ignore
        event_id=event_id,
        user_id=current_user.id,
        registration_id=registration.id,
        overall_rating=int(request.form['overall_rating']),
        content_rating=int(request.form.get('content_rating', 0)) or None,
        organization_rating=int(request.form.get('organization_rating', 0)) or None,
        venue_rating=int(request.form.get('venue_rating', 0)) or None,
        value_rating=int(request.form.get('value_rating', 0)) or None,
        positive_feedback=request.form.get('positive_feedback', '').strip() or None,
        improvement_suggestions=request.form.get('improvement_suggestions', '').strip() or None,
        additional_comments=request.form.get('additional_comments', '').strip() or None,
        would_recommend=request.form.get('would_recommend') == 'yes',
        is_anonymous=request.form.get('is_anonymous') == 'on'
    )
    
    db.session.add(feedback)
    db.session.commit()
    
    flash('Thank you for your feedback! Your input helps us improve future events.', 'success')
    return redirect(url_for('view_feedback', feedback_id=feedback.id))

@app.route('/feedback/<int:feedback_id>')
@login_required
def view_feedback(feedback_id):
    """View submitted feedback"""
    feedback = EventFeedback.query.get_or_404(feedback_id)
    
    # Check if user has permission to view this feedback
    if feedback.user_id != current_user.id and not current_user.is_organizer:
        flash('You can only view your own feedback!', 'error')
        return redirect(url_for('dashboard'))
    
    # If organizer, check if it's their event
    if current_user.is_organizer and feedback.event.organizer_id != current_user.id:
        flash('You can only view feedback for your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('view_feedback.html', feedback=feedback)

@app.route('/event/<int:event_id>/feedback/report')
@login_required
def feedback_report(event_id):
    """Generate feedback summary report for organizers"""
    if not current_user.is_organizer:
        flash('Only organizers can view feedback reports!', 'error')
        return redirect(url_for('dashboard'))
    
    event = Event.query.get_or_404(event_id)
    
    if event.organizer_id != current_user.id:
        flash('You can only view feedback reports for your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    # Get all feedback for the event
    feedback_records = EventFeedback.query.filter_by(event_id=event_id).all()
    
    if not feedback_records:
        flash('No feedback has been submitted for this event yet.', 'info')
        return redirect(url_for('event_detail', event_id=event_id))
    
    # Calculate statistics
    total_feedback = len(feedback_records)
    
    # Average ratings
    avg_overall = sum(f.overall_rating for f in feedback_records) / total_feedback
    avg_content = sum(f.content_rating for f in feedback_records if f.content_rating) / len([f for f in feedback_records if f.content_rating]) if any(f.content_rating for f in feedback_records) else 0
    avg_organization = sum(f.organization_rating for f in feedback_records if f.organization_rating) / len([f for f in feedback_records if f.organization_rating]) if any(f.organization_rating for f in feedback_records) else 0
    avg_venue = sum(f.venue_rating for f in feedback_records if f.venue_rating) / len([f for f in feedback_records if f.venue_rating]) if any(f.venue_rating for f in feedback_records) else 0
    avg_value = sum(f.value_rating for f in feedback_records if f.value_rating) / len([f for f in feedback_records if f.value_rating]) if any(f.value_rating for f in feedback_records) else 0
    
    # Recommendation rate
    recommend_count = sum(1 for f in feedback_records if f.would_recommend)
    recommend_rate = (recommend_count / total_feedback * 100) if total_feedback > 0 else 0
    
    # Rating distribution
    rating_distribution = {}
    for i in range(1, 6):
        rating_distribution[i] = sum(1 for f in feedback_records if f.overall_rating == i)
    
    # Collect all text feedback
    positive_feedback = [f.positive_feedback for f in feedback_records if f.positive_feedback]
    improvement_suggestions = [f.improvement_suggestions for f in feedback_records if f.improvement_suggestions]
    additional_comments = [f.additional_comments for f in feedback_records if f.additional_comments]
    
    return render_template('feedback_report.html',
                         event=event,
                         feedback_records=feedback_records,
                         total_feedback=total_feedback,
                         avg_overall=avg_overall,
                         avg_content=avg_content,
                         avg_organization=avg_organization,
                         avg_venue=avg_venue,
                         avg_value=avg_value,
                         recommend_rate=recommend_rate,
                         rating_distribution=rating_distribution,
                         positive_feedback=positive_feedback,
                         improvement_suggestions=improvement_suggestions,
                         additional_comments=additional_comments)

@app.route('/event/<int:event_id>/feedback/export')
@login_required
def export_feedback(event_id):
    """Export feedback data to CSV"""
    if not current_user.is_organizer:
        flash('Only organizers can export feedback data!', 'error')
        return redirect(url_for('dashboard'))
    
    event = Event.query.get_or_404(event_id)
    
    if event.organizer_id != current_user.id:
        flash('You can only export feedback for your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    feedback_records = EventFeedback.query.filter_by(event_id=event_id).all()
    
    if not feedback_records:
        flash('No feedback data to export.', 'info')
        return redirect(url_for('feedback_report', event_id=event_id))
    
    # Create CSV content
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'User', 'Email', 'Overall Rating', 'Content Rating', 'Organization Rating', 
        'Venue Rating', 'Value Rating', 'Would Recommend', 'Positive Feedback', 
        'Improvement Suggestions', 'Additional Comments', 'Submitted At', 'Anonymous'
    ])
    
    # Write data
    for feedback in feedback_records:
        writer.writerow([
            feedback.user.first_name + ' ' + feedback.user.last_name if not feedback.is_anonymous else 'Anonymous',
            feedback.user.email if not feedback.is_anonymous else 'Anonymous',
            feedback.overall_rating,
            feedback.content_rating or '',
            feedback.organization_rating or '',
            feedback.venue_rating or '',
            feedback.value_rating or '',
            'Yes' if feedback.would_recommend else 'No' if feedback.would_recommend is not None else '',
            feedback.positive_feedback or '',
            feedback.improvement_suggestions or '',
            feedback.additional_comments or '',
            feedback.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
            'Yes' if feedback.is_anonymous else 'No'
        ])
    
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'feedback_{event.id}_{event.date.strftime("%Y%m%d")}.csv'
    )

# Photo Upload and Face Recognition Routes
@app.route('/upload_photo/<int:registration_id>')
@login_required
def upload_photo(registration_id):
    """Photo upload page for face verification"""
    registration = EventRegistration.query.get_or_404(registration_id)
    
    # Check if user has permission to upload photo for this registration
    if registration.user_id != current_user.id:
        flash('You can only upload photos for your own registrations!', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if face embedding already exists
    existing_embedding = FaceEmbedding.query.filter_by(user_id=current_user.id).first()
    
    return render_template('upload_photo.html', registration=registration, existing_embedding=existing_embedding)

@app.route('/upload_photo/<int:registration_id>/submit', methods=['POST'])
@login_required
def submit_photo(registration_id):
    """Handle photo upload and face embedding generation"""
    registration = EventRegistration.query.get_or_404(registration_id)
    
    # Check if user has permission
    if registration.user_id != current_user.id:
        flash('You can only upload photos for your own registrations!', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if file was uploaded
    if 'photo' not in request.files:
        flash('No photo file selected!', 'error')
        return redirect(url_for('upload_photo', registration_id=registration_id))
    
    file = request.files['photo']
    
    if file.filename == '':
        flash('No photo file selected!', 'error')
        return redirect(url_for('upload_photo', registration_id=registration_id))
    
    if file and allowed_file(file.filename):
        # Generate secure filename
        filename = secure_filename(f"{current_user.id}_{registration_id}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file
        file.save(filepath)
        
        # Process face embedding
        success, message = save_face_embedding(current_user.id, filepath, registration_id)
        
        if success:
            flash('Photo uploaded and face embedding created successfully!', 'success')
            return redirect(url_for('view_qr_code', registration_id=registration_id))
        else:
            # Delete the uploaded file if face detection failed
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Photo upload failed: {message}', 'error')
            return redirect(url_for('upload_photo', registration_id=registration_id))
    else:
        flash('Invalid file type. Please upload a PNG, JPG, JPEG, or GIF image.', 'error')
        return redirect(url_for('upload_photo', registration_id=registration_id))

@app.route('/face_verification_scanner')
@login_required
def face_verification_scanner():
    """Face verification scanner interface for security guards"""
    if not current_user.is_organizer:
        flash('Only organizers can access the face verification scanner!', 'error')
        return redirect(url_for('dashboard'))
    
    # Get events organized by current user
    events = Event.query.filter_by(organizer_id=current_user.id, status='active').all()
    
    return render_template('scan_qr.html', events=events)

@app.route('/face_verification_scanner/event/<int:event_id>')
@login_required
def event_face_scanner(event_id):
    """Event-specific face verification scanner"""
    if not current_user.is_organizer:
        flash('Only organizers can access the face verification scanner!', 'error')
        return redirect(url_for('dashboard'))
    
    event = Event.query.get_or_404(event_id)
    
    # Check if current user is the organizer
    if event.organizer_id != current_user.id:
        flash('You can only scan faces for your own events!', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('event_face_scanner.html', event=event)

@app.route('/api/verify_face', methods=['POST'])
@login_required
def verify_face():
    """API endpoint for face verification"""
    if not current_user.is_organizer:
        return jsonify({'success': False, 'error': 'Only organizers can verify faces'}), 403
    
    try:
        data = request.get_json()
        qr_data = data.get('qr_data')
        face_image_data = data.get('face_image')  # Base64 encoded image
        event_id = data.get('event_id')
        
        if not qr_data or not face_image_data:
            return jsonify({'success': False, 'error': 'Missing QR code or face image data'}), 400
        
        # Parse QR code data
        import json
        try:
            qr_info = json.loads(qr_data)
        except json.JSONDecodeError:
            return jsonify({'success': False, 'error': 'Invalid QR code format'}), 400
        
        # Find the registration
        registration = EventRegistration.query.get(qr_info['registration_id'])
        if not registration:
            return jsonify({'success': False, 'error': 'Registration not found'}), 404
        
        # Verify event matches
        if event_id and int(qr_info['event_id']) != event_id:
            return jsonify({'success': False, 'error': 'QR code is for a different event'}), 400
        
        # Get stored face embedding
        face_embedding = FaceEmbedding.query.filter_by(user_id=registration.user_id).first()
        if not face_embedding:
            return jsonify({'success': False, 'error': 'No face data found for this user'}), 404
        
        # Decode face image
        face_image_bytes = base64.b64decode(face_image_data.split(',')[1])
        
        # Save temporary image
        temp_image_path = os.path.join(UPLOAD_FOLDER, f"temp_{current_user.id}_{datetime.now().timestamp()}.jpg")
        with open(temp_image_path, 'wb') as f:
            f.write(face_image_bytes)
        
        try:
            # Detect face in live image
            live_face_encoding, face_detected = detect_and_encode_face(temp_image_path)
            
            if not face_detected or live_face_encoding is None:
                return jsonify({'success': False, 'error': 'No face detected in the live image'}), 400
            
            # Load stored face features
            stored_face_features = pickle.loads(face_embedding.face_embedding)
            
            # Compare faces
            similarity = compare_faces(stored_face_features, live_face_encoding)
            
            # Determine if match is sufficient
            is_match = similarity >= (FACE_MATCH_THRESHOLD * 100)
            
            # Mark face as verified if match is successful
            if is_match:
                face_embedding.is_verified = True
                db.session.commit()
            
            return jsonify({
                'success': True,
                'is_match': is_match,
                'similarity': round(similarity, 2),
                'threshold': FACE_MATCH_THRESHOLD * 100,
                'user_info': {
                    'name': f"{registration.user.first_name} {registration.user.last_name}",
                    'event_title': registration.event.title,
                    'verification_code': qr_info['verification_code']
                }
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Face verification failed: {str(e)}'}), 500

@app.route('/scan_qr/<string:qr_text>', methods=['GET'])
@login_required
def scan_qr(qr_text):
    """Verify QR code and return attendee details"""
    # 1. Find QR code in DB
    qr_code = QRCode.query.filter_by(qr_text=qr_text).first()
    if not qr_code:
        return jsonify({"success": False, "message": "QR Code not found"}), 404

    # 2. Find registration and user
    registration = EventRegistration.query.get(qr_code.registration_id)
    if not registration:
        return jsonify({"success": False, "message": "Registration not found"}), 404

    user = User.query.get(registration.user_id)
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    # 3. Return attendee info
    return jsonify({
        "success": True,
        "user": {
            "name": f"{user.first_name} {user.last_name}",
            "course": getattr(user, "course", "Not provided"),
            "email": user.email,
            "image_url": url_for('static', filename=f'uploads/faces/{user.face_image}', _external=True) if getattr(user, "face_image", None) else None
        }
    })


# Admin Routes for User Management
@app.route('/admin/users')
@login_required
def admin_users():
    """Admin page to manage all users"""
    if not current_user.can_manage_users():
        flash('You are not authorized to access this page!', 'error')
        return redirect(url_for('dashboard'))
    
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    role_filter = request.args.get('role', '')
    
    query = User.query
    
    if search:
        query = query.filter(
            User.username.contains(search) | 
            User.email.contains(search) |
            User.first_name.contains(search) |
            User.last_name.contains(search)
        )
    
    if role_filter:
        query = query.filter_by(role=role_filter)
    
    users = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    
    roles = ['admin', 'organizer', 'checker', 'user']
    return render_template('admin_users.html', users=users, roles=roles, current_filter=role_filter, search=search)

@app.route('/admin/users/<int:user_id>/update_role', methods=['POST'])
@login_required
def update_user_role(user_id):
    """Update a user's role"""
    if not current_user.can_manage_users():
        flash('You are not authorized to perform this action!', 'error')
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    
    if new_role not in ['admin', 'organizer', 'checker', 'user']:
        flash('Invalid role specified!', 'error')
        return redirect(url_for('admin_users'))
    
    # Prevent admin from removing their own admin role
    if user.id == current_user.id and new_role != 'admin':
        flash('You cannot remove your own admin privileges!', 'error')
        return redirect(url_for('admin_users'))
    
    old_role = user.role
    user.role = new_role
    
    # Update is_organizer flag for backward compatibility
    if new_role == 'organizer':
        user.is_organizer = True
    elif old_role == 'organizer' and new_role != 'organizer':
        user.is_organizer = False
    
    db.session.commit()
    
    flash(f'Successfully updated {user.username}\'s role from {old_role} to {new_role}!', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    """Delete a user (admin only)"""
    if not current_user.can_manage_users():
        flash('You are not authorized to perform this action!', 'error')
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    
    # Prevent admin from deleting themselves
    if user.id == current_user.id:
        flash('You cannot delete your own account!', 'error')
        return redirect(url_for('admin_users'))
    
    username = user.username
    db.session.delete(user)
    db.session.commit()
    
    flash(f'Successfully deleted user {username}!', 'success')
    return redirect(url_for('admin_users'))


import os

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    
    port = int(os.environ.get("PORT", 5000))  

   
    app.run(host="0.0.0.0", port=port, debug=True)



