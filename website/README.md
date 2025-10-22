# SignXpress Web Application

A beautiful, modern web application for learning sign language with AI-powered recognition.

## Features

- **User Authentication**: Sign up and login with elegant forms
- **Dashboard**: Clean, modern interface with progress tracking
- **Three Learning Modules**:
  - Numbers (0-9)
  - Alphabets (A-Z)
  - Words (Common sign language words)
- **Progress Tracking**: Visual progress bars and statistics
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Getting Started

1. Open `index.html` in your web browser
2. Create a new account or sign in with existing credentials
3. Explore the three learning modules on the dashboard
4. Track your progress as you learn

## File Structure

```
website/
├── index.html      # Main HTML file with all pages
├── styles.css      # Elegant CSS styling
├── script.js       # JavaScript functionality
└── README.md       # This file
```

## Design Features

- **Color Scheme**: Elegant purple gradient (#667eea to #764ba2)
- **Typography**: Inter font family for modern readability
- **Animations**: Smooth transitions and hover effects
- **Glass Morphism**: Modern frosted glass effects
- **Responsive**: Mobile-first design approach

## Integration with ML Models

The website is designed to integrate with your existing machine learning models:
- `models/number_model.h5` for number recognition
- `models/alphabet_model.h5` for alphabet recognition
- Future word recognition model

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Next Steps

1. Integrate with your Python ML models using Flask/FastAPI
2. Add real-time camera integration for sign recognition
3. Implement user-specific progress persistence
4. Add more learning modules and content
