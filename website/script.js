// SignXpress Web Application JavaScript

class SignXpressApp {
    constructor() {
        this.currentUser = null;
        this.userProgress = {
            numbers: 0,
            alphabets: 0,
            words: 0
        };
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkAuthStatus();
        this.loadUserProgress();
    }

    bindEvents() {
        // Authentication form events
        document.getElementById('loginForm').addEventListener('submit', (e) => this.handleLogin(e));
        document.getElementById('signupForm').addEventListener('submit', (e) => this.handleSignup(e));
        
        // Navigation events
        document.getElementById('showSignup').addEventListener('click', (e) => this.showSignup(e));
        document.getElementById('showLogin').addEventListener('click', (e) => this.showLogin(e));
        document.getElementById('logoutBtn').addEventListener('click', () => this.handleLogout());
        
        // Module events
        document.querySelectorAll('.btn-module').forEach(btn => {
            btn.addEventListener('click', (e) => this.startModule(e));
        });
    }

    checkAuthStatus() {
        const savedUser = localStorage.getItem('signxpress_user');
        if (savedUser) {
            this.currentUser = JSON.parse(savedUser);
            this.showDashboard();
        } else {
            this.showLogin();
        }
    }

    showLogin(e) {
        if (e) e.preventDefault();
        document.getElementById('loginPage').classList.remove('hidden');
        document.getElementById('signupPage').classList.add('hidden');
        document.getElementById('dashboardPage').classList.add('hidden');
    }

    showSignup(e) {
        if (e) e.preventDefault();
        document.getElementById('signupPage').classList.remove('hidden');
        document.getElementById('loginPage').classList.add('hidden');
        document.getElementById('dashboardPage').classList.add('hidden');
    }

    showDashboard() {
        document.getElementById('dashboardPage').classList.remove('hidden');
        document.getElementById('loginPage').classList.add('hidden');
        document.getElementById('signupPage').classList.add('hidden');
        
        if (this.currentUser) {
            document.getElementById('userName').textContent = this.currentUser.name;
            this.updateProgress();
        }
    }

    handleLogin(e) {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;

        // Simple validation
        if (!email || !password) {
            this.showNotification('Please fill in all fields', 'error');
            return;
        }

        // Simulate login process
        this.showLoading(true);
        
        setTimeout(() => {
            // For demo purposes, accept any email/password combination
            this.currentUser = {
                name: email.split('@')[0],
                email: email,
                joinDate: new Date().toISOString()
            };
            
            localStorage.setItem('signxpress_user', JSON.stringify(this.currentUser));
            this.showLoading(false);
            this.showNotification('Welcome back!', 'success');
            this.showDashboard();
        }, 1000);
    }

    handleSignup(e) {
        e.preventDefault();
        const name = document.getElementById('signupName').value;
        const email = document.getElementById('signupEmail').value;
        const password = document.getElementById('signupPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        // Validation
        if (!name || !email || !password || !confirmPassword) {
            this.showNotification('Please fill in all fields', 'error');
            return;
        }

        if (password !== confirmPassword) {
            this.showNotification('Passwords do not match', 'error');
            return;
        }

        if (password.length < 6) {
            this.showNotification('Password must be at least 6 characters', 'error');
            return;
        }

        // Simulate signup process
        this.showLoading(true);
        
        setTimeout(() => {
            this.currentUser = {
                name: name,
                email: email,
                joinDate: new Date().toISOString()
            };
            
            localStorage.setItem('signxpress_user', JSON.stringify(this.currentUser));
            this.showLoading(false);
            this.showNotification('Account created successfully!', 'success');
            this.showDashboard();
        }, 1000);
    }

    handleLogout() {
        localStorage.removeItem('signxpress_user');
        localStorage.removeItem('signxpress_progress');
        this.currentUser = null;
        this.userProgress = { numbers: 0, alphabets: 0, words: 0 };
        this.showLogin();
        this.showNotification('Logged out successfully', 'success');
    }

    startModule(e) {
        const moduleCard = e.target.closest('.module-card');
        const module = moduleCard.dataset.module;
        
        this.showNotification(`Starting ${module} module...`, 'info');
        
        // Here you would integrate with your ML models
        // For now, we'll simulate progress
        setTimeout(() => {
            this.userProgress[module] = Math.min(this.userProgress[module] + 10, 100);
            this.saveUserProgress();
            this.updateProgress();
            this.showNotification(`${module} module progress updated!`, 'success');
        }, 500);
    }

    loadUserProgress() {
        const savedProgress = localStorage.getItem('signxpress_progress');
        if (savedProgress) {
            this.userProgress = JSON.parse(savedProgress);
        }
    }

    saveUserProgress() {
        localStorage.setItem('signxpress_progress', JSON.stringify(this.userProgress));
    }

    updateProgress() {
        Object.keys(this.userProgress).forEach(module => {
            const progress = this.userProgress[module];
            const moduleCard = document.querySelector(`[data-module="${module}"]`);
            
            if (moduleCard) {
                const progressFill = moduleCard.querySelector('.progress-fill');
                const progressText = moduleCard.querySelector('.progress-text');
                
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `${progress}% Complete`;
            }
        });

        // Update stats
        const totalSigns = Object.values(this.userProgress).reduce((sum, progress) => sum + progress, 0);
        const hoursLearned = Math.floor(totalSigns / 10);
        const dayStreak = Math.floor(totalSigns / 20);

        document.querySelectorAll('.stat-number')[0].textContent = hoursLearned;
        document.querySelectorAll('.stat-number')[1].textContent = totalSigns;
        document.querySelectorAll('.stat-number')[2].textContent = dayStreak;
    }

    showLoading(show) {
        const buttons = document.querySelectorAll('.btn-primary');
        buttons.forEach(btn => {
            if (show) {
                btn.disabled = true;
                btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            } else {
                btn.disabled = false;
                btn.innerHTML = btn.id === 'loginForm' ? 'Sign In' : 'Create Account';
            }
        });
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${this.getNotificationColor(type)};
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
            max-width: 300px;
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            info: 'info-circle',
            warning: 'exclamation-triangle'
        };
        return icons[type] || 'info-circle';
    }

    getNotificationColor(type) {
        const colors = {
            success: '#10b981',
            error: '#ef4444',
            info: '#3b82f6',
            warning: '#f59e0b'
        };
        return colors[type] || '#3b82f6';
    }
}

// Add notification animations to CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .notification-content i {
        font-size: 1.1rem;
    }
`;
document.head.appendChild(style);

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new SignXpressApp();
});
