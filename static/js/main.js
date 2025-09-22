document.addEventListener('DOMContentLoaded', function () {
    // ===== Bootstrap Initializations =====
    [...document.querySelectorAll('[data-bs-toggle="tooltip"]')]
        .forEach(el => new bootstrap.Tooltip(el));
    [...document.querySelectorAll('[data-bs-toggle="popover"]')]
        .forEach(el => new bootstrap.Popover(el));

    // Auto-hide alerts after 5 seconds
    document.querySelectorAll('.alert').forEach(alert => {
        setTimeout(() => new bootstrap.Alert(alert).close(), 5000);
    });

    // ===== Search Functionality =====
    const searchInput = document.getElementById('search');
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener('input', function () {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                const form = searchInput.closest('form');
                if (form && searchInput.value.length > 2) form.submit();
            }, 500);
        });
    }

    // ===== Confirmation Dialogs (delegated) =====
    document.body.addEventListener('click', function (e) {
        const link = e.target.closest('a[href*="register_event"], a[href*="cancel_registration"], a[href*="delete_event"]');
        if (link) {
            let msg = '';
            if (link.href.includes('register_event')) msg = 'Are you sure you want to register for this event?';
            else if (link.href.includes('cancel_registration')) msg = 'Are you sure you want to cancel your registration? This action cannot be undone.';
            else if (link.href.includes('delete_event')) msg = 'Are you sure you want to delete this event? This action cannot be undone and will cancel all registrations.';
            if (msg && !confirm(msg)) e.preventDefault();
        }
    });

    // ===== Date & Time Validation =====
    document.querySelectorAll('input[type="date"]').forEach(input => {
        const today = new Date().toISOString().split('T')[0];
        input.setAttribute('min', today);
        input.addEventListener('change', function () {
            const selectedDate = new Date(this.value);
            const today = new Date(); today.setHours(0,0,0,0);
            this.setCustomValidity(selectedDate < today ? 'Event date cannot be in the past' : '');
        });
    });

    document.querySelectorAll('input[type="time"]').forEach(input => {
        input.addEventListener('change', function () {
            const selectedTime = this.value;
            const dateInput = this.closest('form')?.querySelector('input[type="date"]');
            if (dateInput?.value) {
                const selectedDate = new Date(dateInput.value);
                const today = new Date(); today.setHours(0,0,0,0);
                if (selectedDate.getTime() === today.getTime()) {
                    const now = new Date();
                    const [h, m] = selectedTime.split(':');
                    selectedDate.setHours(+h,+m,0,0);
                    this.setCustomValidity(selectedDate <= now ? 'Event time must be in the future' : '');
                } else this.setCustomValidity('');
            }
        });
    });

    // ===== Price & Max Attendees Validation =====
    document.querySelectorAll('input[name="price"]').forEach(input => {
        input.addEventListener('input', function () {
            this.setCustomValidity(this.value < 0 ? 'Price cannot be negative' : '');
        });
    });
    document.querySelectorAll('input[name="max_attendees"]').forEach(input => {
        input.addEventListener('input', function () {
            this.setCustomValidity(this.value < 1 ? 'Maximum attendees must be at least 1' : '');
        });
    });

    // ===== Smooth Scroll =====
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener('click', function (e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) { e.preventDefault(); target.scrollIntoView({behavior:'smooth', block:'start'}); }
        });
    });

    // ===== Character Counter =====
    document.querySelectorAll('textarea[maxlength]').forEach(textarea => {
        const maxLength = textarea.getAttribute('maxlength');
        const counter = document.createElement('small');
        counter.className = 'text-muted float-end';
        textarea.parentNode.appendChild(counter);
        function updateCounter() {
            const remaining = maxLength - textarea.value.length;
            counter.textContent = `${remaining} characters remaining`;
            counter.className = remaining < 50 ? 'text-danger float-end' : 'text-muted float-end';
        }
        textarea.addEventListener('input', updateCounter);
        updateCounter();
    });

    // ===== Lazy Loading for Images =====
    const images = document.querySelectorAll('img[data-src]');
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    obs.unobserve(img);
                }
            });
        });
        images.forEach(img => observer.observe(img));
    } else images.forEach(img => img.src = img.dataset.src);

    // ===== Back to Top Button =====
    const backToTop = document.createElement('button');
    backToTop.innerHTML = '<i class="fas fa-arrow-up"></i>';
    backToTop.className = 'btn btn-primary position-fixed';
    backToTop.style.cssText = 'bottom:20px; right:20px; z-index:1000; display:none; border-radius:50%; width:50px; height:50px;';
    document.body.appendChild(backToTop);
    window.addEventListener('scroll', () => backToTop.style.display = window.pageYOffset > 300 ? 'block' : 'none');
    backToTop.addEventListener('click', () => window.scrollTo({top:0, behavior:'smooth'}));

    // ===== Navbar Scroll Shadow / Shrink =====
    const navbar = document.querySelector('.navbar-glass');
    const onScrollNavbar = () => {
        if (!navbar) return;
        if (window.pageYOffset > 10) navbar.classList.add('navbar-scrolled');
        else navbar.classList.remove('navbar-scrolled');
    };
    onScrollNavbar();
    window.addEventListener('scroll', onScrollNavbar, { passive: true });

    // ===== Dark Mode Toggle (with persistence) =====
    const themeToggle = document.getElementById('themeToggle');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('eventhub-theme');
    const applyTheme = (mode) => {
        document.body.classList.toggle('dark-mode', mode === 'dark');
        if (themeToggle) themeToggle.innerHTML = `<i class="fas ${mode==='dark' ? 'fa-sun' : 'fa-moon'}"></i>`;
    };
    applyTheme(savedTheme || (prefersDark ? 'dark' : 'light'));
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const next = document.body.classList.contains('dark-mode') ? 'light' : 'dark';
            localStorage.setItem('eventhub-theme', next);
            applyTheme(next);
        });
    }

    // ===== Keyboard Shortcuts =====
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k' && searchInput) { e.preventDefault(); searchInput.focus(); }
        if (e.key === 'Escape') document.querySelectorAll('.alert').forEach(alert => new bootstrap.Alert(alert).close());
    });

    // ===== Print =====
    document.querySelectorAll('[data-print]').forEach(button => {
        button.addEventListener('click', function () {
            const target = document.querySelector(this.dataset.print);
            if (target) {
                const printWindow = window.open('', '_blank');
                printWindow.document.write(`
                    <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>Print - ${document.title}</title>
                            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                        </head>
                        <body>${target.innerHTML}</body>
                    </html>
                `);
                printWindow.document.close();
                printWindow.print();
            }
        });
    });

    // ===== Password Toggle Functionality =====
    document.querySelectorAll('[id^="togglePassword"], [id^="toggleConfirmPassword"]').forEach(toggleBtn => {
        toggleBtn.addEventListener('click', function () {
            const inputId = this.id === 'togglePassword' ? 'password' : 'confirm_password';
            const passwordInput = document.getElementById(inputId);
            const icon = this.querySelector('i');
            
            if (passwordInput.type === 'password') {
                passwordInput.type = 'text';
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
            } else {
                passwordInput.type = 'password';
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
            }
        });
    });
});

// ===== Utility Functions =====
window.EventHub = {
    formatDate: dateString => new Date(dateString).toLocaleDateString('en-US', {year:'numeric', month:'long', day:'numeric'}),
    formatTime: timeString => {
        const [h, m] = timeString.split(':'); const d = new Date(); d.setHours(+h,+m); return d.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit',hour12:true});
    },
    formatCurrency: amount => new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(amount)
};

document.addEventListener('DOMContentLoaded', function () {
    const roleSelect = document.getElementById('role');
    const studentOptions = document.getElementById('student-options');
    const facultyOptions = document.getElementById('faculty-options');
    const technicianOptions = document.getElementById('technician-options');

    if (roleSelect) {
        roleSelect.addEventListener('change', function () {
            studentOptions.classList.add('d-none');
            facultyOptions.classList.add('d-none');
            technicianOptions.classList.add('d-none');

            if (this.value === 'student') {
                studentOptions.classList.remove('d-none');
            } else if (this.value === 'faculty') {
                facultyOptions.classList.remove('d-none');
            } else if (this.value === 'technician') {
                technicianOptions.classList.remove('d-none');
            }
        });
    }
});
