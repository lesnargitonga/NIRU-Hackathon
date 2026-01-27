import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Settings as SettingsIcon, Save, RefreshCw, AlertTriangle } from 'lucide-react';

function Settings() {
  const [settings, setSettings] = useState({
    // Drone Settings
    maxAltitude: 120,
    maxSpeed: 15,
    batteryWarningLevel: 20,
    batteryCriticalLevel: 5,
    autoLandBattery: 10,
    
    // System Settings
    updateRate: 10,
    logLevel: 'INFO',
    enableWeatherCheck: true,
    enableCollisionAvoidance: true,
    
    // API Settings
    apiHost: 'localhost',
    apiPort: 5000,
    enableSSL: false,
    
    // UI Settings
    darkMode: false,
    mapProvider: 'openstreetmap',
    defaultZoom: 12,
    showFlightPaths: true,
    enableNotifications: true
  });

  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Map between UI settings and backend config.json
  const mapConfigToSettings = (cfg) => {
    const drone = (cfg || {}).drone_settings || {};
    const sim = (cfg || {}).simulation_settings || {};
    const api = (cfg || {}).api_settings || {};
    const logging = (cfg || {}).logging || {};
    return {
      maxAltitude: Number(drone.max_altitude ?? 120),
      maxSpeed: Number(drone.max_speed ?? 15),
      batteryWarningLevel: Number(drone.battery_warning_level ?? 20),
      batteryCriticalLevel: Number(drone.battery_critical_level ?? 5),
      autoLandBattery: Number(drone.auto_land_battery ?? 10),
      updateRate: Number(sim.update_rate ?? 10),
      logLevel: String(logging.level ?? 'INFO'),
      enableWeatherCheck: Boolean(sim.weather_simulation ?? true),
      enableCollisionAvoidance: Boolean(sim.collision_detection ?? true),
      apiHost: String(api.host ?? 'localhost'),
      apiPort: Number(api.port ?? 5000),
      enableSSL: Boolean(api.enable_ssl ?? false),
      // UI-only settings remain as-is
      darkMode: settings.darkMode,
      mapProvider: settings.mapProvider,
      defaultZoom: settings.defaultZoom,
      showFlightPaths: settings.showFlightPaths,
      enableNotifications: settings.enableNotifications
    };
  };

  const mapSettingsToConfig = (currentCfg, ui) => {
    const cfg = { ...(currentCfg || {}) };
    cfg.drone_settings = {
      ...(cfg.drone_settings || {}),
      max_speed: Number(ui.maxSpeed),
      max_altitude: Number(ui.maxAltitude),
      battery_warning_level: Number(ui.batteryWarningLevel),
      battery_critical_level: Number(ui.batteryCriticalLevel),
      auto_land_battery: Number(ui.autoLandBattery),
    };
    cfg.simulation_settings = {
      ...(cfg.simulation_settings || {}),
      update_rate: Number(ui.updateRate),
      weather_simulation: Boolean(ui.enableWeatherCheck),
      collision_detection: Boolean(ui.enableCollisionAvoidance),
    };
    cfg.api_settings = {
      ...(cfg.api_settings || {}),
      host: String(ui.apiHost),
      port: Number(ui.apiPort),
      enable_ssl: Boolean(ui.enableSSL),
    };
    cfg.logging = {
      ...(cfg.logging || {}),
      level: String(ui.logLevel || 'INFO'),
    };
    return cfg;
  };

  // Load config.json on mount
  useEffect(() => {
    const loadConfig = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await axios.get('/api/config');
        if (res.data?.success && res.data.config) {
          setSettings(prev => mapConfigToSettings(res.data.config));
        } else {
          setError('Failed to load config');
        }
      } catch (e) {
        setError('Backend unavailable or /api/config error');
      } finally {
        setLoading(false);
      }
    };
    loadConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSave = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch current config for merge
      const current = await axios.get('/api/config');
      const merged = mapSettingsToConfig(current.data?.config || {}, settings);
      const res = await axios.post('/api/config', { config: merged });
      if (res.data?.success) {
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
      } else {
        setError('Failed to save config');
      }
    } catch (e) {
      setError('Save failed: backend unavailable');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (window.confirm('Reset all settings to defaults? This cannot be undone.')) {
      // Reset to default values
      setSettings({
        maxAltitude: 120,
        maxSpeed: 15,
        batteryWarningLevel: 20,
        batteryCriticalLevel: 5,
        autoLandBattery: 10,
        updateRate: 10,
        logLevel: 'INFO',
        enableWeatherCheck: true,
        enableCollisionAvoidance: true,
        apiHost: 'localhost',
        apiPort: 5000,
        enableSSL: false,
        darkMode: false,
        mapProvider: 'openstreetmap',
        defaultZoom: 12,
        showFlightPaths: true,
        enableNotifications: true
      });
    }
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center">
          <SettingsIcon className="h-7 w-7 mr-2" />
          Settings
        </h1>
        <p className="mt-1 text-sm text-gray-600">
          Configure system parameters and preferences
        </p>
      </div>

      {saved && (
        <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <div className="h-5 w-5 text-green-400">✓</div>
            <p className="ml-3 text-sm text-green-700">Settings saved successfully!</p>
          </div>
        </div>
      )}

      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Drone Settings */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold mb-4">Drone Parameters</h2>
          <div className="space-y-4">
            <SettingItem
              label="Maximum Altitude (m)"
              value={settings.maxAltitude}
              onChange={(value) => handleChange('maxAltitude', value)}
              type="number"
              min="1"
              max="400"
            />
            
            <SettingItem
              label="Maximum Speed (m/s)"
              value={settings.maxSpeed}
              onChange={(value) => handleChange('maxSpeed', value)}
              type="number"
              min="1"
              max="30"
            />
            
            <SettingItem
              label="Battery Warning Level (%)"
              value={settings.batteryWarningLevel}
              onChange={(value) => handleChange('batteryWarningLevel', value)}
              type="number"
              min="10"
              max="50"
            />
            
            <SettingItem
              label="Battery Critical Level (%)"
              value={settings.batteryCriticalLevel}
              onChange={(value) => handleChange('batteryCriticalLevel', value)}
              type="number"
              min="1"
              max="20"
            />
            
            <SettingItem
              label="Auto-Land Battery Level (%)"
              value={settings.autoLandBattery}
              onChange={(value) => handleChange('autoLandBattery', value)}
              type="number"
              min="1"
              max="25"
            />
          </div>
        </div>

        {/* System Settings */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold mb-4">System Configuration</h2>
          <div className="space-y-4">
            <SettingItem
              label="Update Rate (Hz)"
              value={settings.updateRate}
              onChange={(value) => handleChange('updateRate', value)}
              type="number"
              min="1"
              max="100"
            />
            
            <SettingItem
              label="Log Level"
              value={settings.logLevel}
              onChange={(value) => handleChange('logLevel', value)}
              type="select"
              options={['DEBUG', 'INFO', 'WARNING', 'ERROR']}
            />
            
            <ToggleItem
              label="Enable Weather Checking"
              checked={settings.enableWeatherCheck}
              onChange={(checked) => handleChange('enableWeatherCheck', checked)}
            />
            
            <ToggleItem
              label="Enable Collision Avoidance"
              checked={settings.enableCollisionAvoidance}
              onChange={(checked) => handleChange('enableCollisionAvoidance', checked)}
            />
          </div>
        </div>

        {/* API Settings */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold mb-4">API Configuration</h2>
          <div className="space-y-4">
            <SettingItem
              label="API Host"
              value={settings.apiHost}
              onChange={(value) => handleChange('apiHost', value)}
              type="text"
            />
            
            <SettingItem
              label="API Port"
              value={settings.apiPort}
              onChange={(value) => handleChange('apiPort', value)}
              type="number"
              min="1000"
              max="65535"
            />
            
            <ToggleItem
              label="Enable SSL/HTTPS"
              checked={settings.enableSSL}
              onChange={(checked) => handleChange('enableSSL', checked)}
            />
          </div>
        </div>

        {/* UI Settings */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold mb-4">User Interface</h2>
          <div className="space-y-4">
            <ToggleItem
              label="Dark Mode"
              checked={settings.darkMode}
              onChange={(checked) => handleChange('darkMode', checked)}
            />
            
            <SettingItem
              label="Map Provider"
              value={settings.mapProvider}
              onChange={(value) => handleChange('mapProvider', value)}
              type="select"
              options={['openstreetmap', 'satellite', 'terrain']}
            />
            
            <SettingItem
              label="Default Map Zoom"
              value={settings.defaultZoom}
              onChange={(value) => handleChange('defaultZoom', value)}
              type="number"
              min="1"
              max="20"
            />
            
            <ToggleItem
              label="Show Flight Paths"
              checked={settings.showFlightPaths}
              onChange={(checked) => handleChange('showFlightPaths', checked)}
            />
            
            <ToggleItem
              label="Enable Notifications"
              checked={settings.enableNotifications}
              onChange={(checked) => handleChange('enableNotifications', checked)}
            />
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="mt-8 flex justify-between">
        <button
          onClick={handleReset}
          className="btn-secondary flex items-center"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Reset to Defaults
        </button>
        
        <button
          onClick={handleSave}
          className={`btn-primary flex items-center ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={loading}
        >
          <Save className="h-4 w-4 mr-2" />
          {loading ? 'Saving...' : 'Save Settings'}
        </button>
      </div>

      {/* Warning */}
      <div className="mt-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-start">
          <AlertTriangle className="h-5 w-5 text-yellow-400 mt-0.5" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-yellow-800">Important</h3>
            <p className="mt-1 text-sm text-yellow-700">
              Changes to drone parameters may affect flight safety. Ensure all values are within safe operating limits 
              before applying changes. Some settings may require a system restart to take effect.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function SettingItem({ label, value, onChange, type = 'text', options = [], ...props }) {
  if (type === 'select') {
    return (
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}
        </label>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        {...props}
      />
    </div>
  );
}

function ToggleItem({ label, checked, onChange }) {
  return (
    <div className="flex items-center justify-between">
      <label className="text-sm font-medium text-gray-700">
        {label}
      </label>
      <button
        onClick={() => onChange(!checked)}
        className={`
          relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent 
          transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
          ${checked ? 'bg-blue-600' : 'bg-gray-200'}
        `}
      >
        <span
          className={`
            pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 
            transition duration-200 ease-in-out
            ${checked ? 'translate-x-5' : 'translate-x-0'}
          `}
        />
      </button>
    </div>
  );
}

export default Settings;
