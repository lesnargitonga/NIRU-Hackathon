# Lesnar AI Development Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with pip
- Node.js 16+ with npm
- Git (optional but recommended)

### Installation

1. **Clone or download the project**
   ```bash
   # If using Git
   git clone <repository-url>
   cd "Lesnar AI"
   ```

2. **Run the setup script**
   ```bash
   # Windows
   setup.bat
   
   # Manual setup (all platforms)
   pip install -r backend/requirements.txt
   pip install -r ai_modules/requirements.txt
   cd frontend && npm install && cd ..
   ```

### Running the System

#### Option 1: Start All Services (Windows)
```bash
start_all.bat
```

#### Option 2: Start Individual Services
```bash
# Terminal 1: Backend API
start_backend.bat
# or: cd backend && python app.py

# Terminal 2: Frontend Dashboard  
start_frontend.bat
# or: cd frontend && npm start

# Terminal 3: Simulation (Optional)
start_simulation.bat
# or: cd drone_simulation && python main.py
```

#### Access the System
- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000 (shows available endpoints)

---

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │  Simulation     │
│   (React)       │◄──►│   (Flask)       │◄──►│   (Python)      │
│   Port: 3000    │    │   Port: 5000    │    │   Standalone    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     AI Modules            │
                    │  - Computer Vision        │
                    │  - Swarm Intelligence     │
                    │  - Shared Utilities       │
                    └───────────────────────────┘
```

### Data Flow
1. **Drone Simulation** generates telemetry data
2. **Backend API** receives and processes drone data
3. **Frontend Dashboard** displays real-time information
4. **AI Modules** provide intelligent decision making
5. **WebSocket** connections enable real-time updates

---

## 📁 Project Structure

```
Lesnar AI/
├── backend/                    # Flask API Server
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # Backend documentation
│
├── frontend/                   # React Dashboard
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── context/          # State management
│   │   └── App.js           # Main React app
│   ├── package.json         # Node dependencies
│   └── README.md           # Frontend documentation
│
├── drone_simulation/          # Drone Physics & Control
│   ├── simulator.py          # Advanced drone simulator
│   ├── main.py              # Simulation entry point
│   └── README.md           # Simulation documentation
│
├── ai_modules/               # AI/ML Components
│   ├── computer_vision.py   # Object detection & tracking
│   ├── swarm_intelligence.py # Multi-drone coordination
│   └── requirements.txt     # AI dependencies
│
├── shared/                   # Shared Utilities
│   └── utils.py             # Common functions & constants
│
├── docs/                    # Documentation
│   ├── API.md              # API documentation
│   ├── FEATURES.md         # Feature specifications
│   └── DEPLOYMENT.md       # Production deployment
│
├── config.json             # System configuration
├── setup.bat              # Windows setup script
├── start_all.bat          # Start all services
└── README.md             # Main project documentation
```

---

## 🔧 Development Workflow

### Adding New Features

1. **Backend API Endpoint**
   ```python
   # In backend/app.py
   @app.route('/api/new-feature', methods=['POST'])
   def new_feature():
       # Implementation
       return jsonify({'success': True})
   ```

2. **Frontend Component**
   ```javascript
   // In frontend/src/components/NewFeature.js
   import React from 'react';
   
   function NewFeature() {
       return <div>New Feature Component</div>;
   }
   
   export default NewFeature;
   ```

3. **Drone Simulation Enhancement**
   ```python
   # In drone_simulation/simulator.py
   class DroneSimulator:
       def new_capability(self):
           # Implementation
           pass
   ```

### Code Standards

- **Python**: PEP 8 style guide
- **JavaScript**: ESLint with React rules
- **Comments**: Docstrings for functions, inline for complex logic
- **Testing**: Unit tests for critical functions

---

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm test
```

### Integration Testing
```bash
# Start all services then run
python tests/integration_tests.py
```

---

## 🔌 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/drones` | List all drones |
| POST | `/api/drones` | Create new drone |
| GET | `/api/drones/{id}` | Get drone details |
| DELETE | `/api/drones/{id}` | Remove drone |
| POST | `/api/drones/{id}/arm` | Arm drone |
| POST | `/api/drones/{id}/takeoff` | Takeoff drone |
| POST | `/api/drones/{id}/land` | Land drone |
| POST | `/api/drones/{id}/goto` | Navigate drone |
| POST | `/api/emergency` | Emergency land all |

### WebSocket Events
- `telemetry_update`: Real-time drone data
- `alert`: System alerts and warnings
- `mission_status`: Mission progress updates

---

## 🚀 Advanced Features

### Computer Vision Integration
```python
from ai_modules.computer_vision import ComputerVisionSystem

cv_system = ComputerVisionSystem()
result = cv_system.process_frame(camera_image)
```

### Swarm Intelligence
```python
from ai_modules.swarm_intelligence import SwarmIntelligence

swarm = SwarmIntelligence()
swarm.add_drone(drone_data)
commands = swarm.get_coordination_commands()
```

### Custom Missions
```python
from drone_simulation.simulator import Mission

mission = Mission(
    waypoints=[(lat1, lon1, alt1), (lat2, lon2, alt2)],
    mission_type="SURVEILLANCE",
    estimated_duration=600
)
```

---

## 🔧 Configuration

Edit `config.json` to customize system behavior:

```json
{
  "drone_settings": {
    "max_speed": 15.0,
    "max_altitude": 120.0,
    "battery_warning_level": 20.0
  },
  "api_settings": {
    "port": 5000,
    "debug": false
  }
}
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```
   Error: Port 5000 is already in use
   Solution: Change port in config.json or kill existing process
   ```

2. **Module Not Found**
   ```
   Error: No module named 'flask'
   Solution: Run pip install -r backend/requirements.txt
   ```

3. **Frontend Build Errors**
   ```
   Error: npm command not found
   Solution: Install Node.js from nodejs.org
   ```

### Debug Mode
```bash
# Enable debug mode
cd backend
FLASK_DEBUG=1 python app.py
```

---

## 📈 Performance Optimization

- **Database**: Consider PostgreSQL for production
- **Caching**: Implement Redis for real-time data
- **Load Balancing**: Use Nginx for multiple instances
- **Monitoring**: Add Prometheus/Grafana metrics

---

## 🔒 Security Considerations

- API authentication and authorization
- Input validation and sanitization
- HTTPS encryption for production
- Rate limiting for API endpoints
- Secure WebSocket connections

---

## 📞 Support

For questions or issues:
1. Check this documentation
2. Review code comments
3. Create an issue in the repository
4. Contact the development team

---

*Last updated: September 2025*
*Lesnar AI Ltd. - Advanced Drone Automation*
