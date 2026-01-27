import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import io from 'socket.io-client';
import './App.css';

import { BACKEND_URL } from './config';

// Components
import Header from './components/Header';
import HealthStatusIndicator from './components/HealthStatusIndicator';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import DroneMap from './components/DroneMap';
import DroneList from './components/DroneList';
import MissionControl from './components/MissionControl';
import Analytics from './components/Analytics';
import Settings from './components/Settings';

// Context for global state
import { DroneProvider } from './context/DroneContext';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [connected, setConnected] = useState(false);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io(BACKEND_URL);
    
    newSocket.on('connect', () => {
      console.log('Connected to Lesnar AI backend');
      setConnected(true);
      
      // Subscribe to telemetry updates
      newSocket.emit('subscribe_telemetry');
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from backend');
      setConnected(false);
    });

    newSocket.on('telemetry_update', (data) => {
      // Handle real-time telemetry updates
      console.log('Received telemetry update:', data);
    });

    setSocket(newSocket);

    // Cleanup on unmount
    return () => {
      newSocket.close();
    };
  }, []);

  return (
    <DroneProvider>
      <Router>
        <div className="h-screen bg-gray-900 flex overflow-hidden">
          {/* Sidebar */}
          <Sidebar 
            isOpen={sidebarOpen} 
            onClose={() => setSidebarOpen(false)} 
          />

          {/* Main content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Header */}
            <Header 
              onMenuClick={() => setSidebarOpen(!sidebarOpen)}
              connected={connected}
            />

            {/* Page content */}
            <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100">
              <Routes>
                <Route path="/" element={<Dashboard socket={socket} />} />
                <Route path="/map" element={<DroneMap socket={socket} />} />
                <Route path="/drones" element={<DroneList socket={socket} />} />
                <Route path="/missions" element={<MissionControl socket={socket} />} />
                <Route path="/analytics" element={<Analytics socket={socket} />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </main>
          </div>
          {/* Global health status */}
          <HealthStatusIndicator />
        </div>
      </Router>
    </DroneProvider>
  );
}

export default App;
