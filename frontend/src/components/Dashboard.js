import React, { useEffect } from 'react';
import { 
  Rocket, 
  Battery, 
  MapPin, 
  Activity,
  AlertTriangle,
  Zap,
  TrendingUp
} from 'lucide-react';
import { useDrones } from '../context/DroneContext';

function Dashboard({ socket }) {
  const { 
    drones, 
    fleetStatus, 
    loading, 
    error, 
    updateTelemetry 
  } = useDrones();

  useEffect(() => {
    if (socket) {
      socket.on('telemetry_update', (data) => {
        updateTelemetry(data);
      });

      return () => {
        socket.off('telemetry_update');
      };
    }
  }, [socket, updateTelemetry]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-600">
          Real-time overview of your drone fleet operations
        </p>
      </div>

      {/* Error display */}
      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <AlertTriangle className="h-5 w-5 text-red-400" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="mt-1 text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Fleet Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatusCard
          title="Total Drones"
          value={fleetStatus.total_drones}
          icon={Rocket}
          color="blue"
        />
        <StatusCard
          title="Armed"
          value={fleetStatus.armed_drones}
          icon={Zap}
          color="yellow"
        />
        <StatusCard
          title="Flying"
          value={fleetStatus.flying_drones}
          icon={TrendingUp}
          color="green"
        />
        <StatusCard
          title="Low Battery"
          value={fleetStatus.low_battery_drones}
          icon={Battery}
          color="red"
        />
      </div>

      {/* Drone Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {drones.map((drone) => (
          <DroneCard key={drone.drone_id} drone={drone} />
        ))}
        
        {drones.length === 0 && (
          <div className="col-span-full bg-gray-50 rounded-lg border-2 border-dashed border-gray-300 p-12 text-center">
            <Rocket className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No drones</h3>
            <p className="mt-1 text-sm text-gray-500">
              Get started by adding your first drone to the fleet.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusCard({ title, value, icon: Icon, color }) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    yellow: 'bg-yellow-50 text-yellow-700 border-yellow-200',
    green: 'bg-green-50 text-green-700 border-green-200',
    red: 'bg-red-50 text-red-700 border-red-200'
  };

  const iconColorClasses = {
    blue: 'text-blue-500',
    yellow: 'text-yellow-500',
    green: 'text-green-500',
    red: 'text-red-500'
  };

  return (
    <div className={`border rounded-lg p-6 ${colorClasses[color]}`}>
      <div className="flex items-center">
        <Icon className={`h-8 w-8 ${iconColorClasses[color]}`} />
        <div className="ml-4">
          <p className="text-sm font-medium opacity-80">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
        </div>
      </div>
    </div>
  );
}

function DroneCard({ drone }) {
  const getStatusColor = (mode, armed, altitude) => {
    if (mode === 'EMERGENCY' || mode === 'CRITICAL') return 'red';
    if (altitude > 1 && armed) return 'green';
    if (armed) return 'yellow';
    return 'gray';
  };

  const getBatteryColor = (battery) => {
    if (battery >= 60) return 'green';
    if (battery >= 30) return 'yellow';
    return 'red';
  };

  const statusColor = getStatusColor(drone.mode, drone.armed, drone.altitude);
  const batteryColor = getBatteryColor(drone.battery);

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          {drone.drone_id}
        </h3>
        <div className={`
          px-2 py-1 rounded-full text-xs font-medium
          ${statusColor === 'green' ? 'bg-green-100 text-green-800' :
            statusColor === 'yellow' ? 'bg-yellow-100 text-yellow-800' :
            statusColor === 'red' ? 'bg-red-100 text-red-800' :
            'bg-gray-100 text-gray-800'
          }
        `}>
          {drone.mode}
        </div>
      </div>

      {/* Status indicators */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="flex items-center">
          <Activity className="h-4 w-4 text-gray-400 mr-2" />
          <span className="text-sm text-gray-600">
            {drone.altitude.toFixed(1)}m
          </span>
        </div>
        <div className="flex items-center">
          <Battery className={`h-4 w-4 mr-2 ${
            batteryColor === 'green' ? 'text-green-500' :
            batteryColor === 'yellow' ? 'text-yellow-500' :
            'text-red-500'
          }`} />
          <span className="text-sm text-gray-600">
            {drone.battery.toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Location */}
      <div className="flex items-center mb-4">
        <MapPin className="h-4 w-4 text-gray-400 mr-2" />
        <span className="text-sm text-gray-600">
          {drone.latitude.toFixed(4)}, {drone.longitude.toFixed(4)}
        </span>
      </div>

      {/* Speed and heading */}
      <div className="flex items-center justify-between text-sm text-gray-600">
        <span>Speed: {drone.speed.toFixed(1)} m/s</span>
        <span>Heading: {drone.heading.toFixed(0)}°</span>
      </div>

      {/* Last update */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          Last update: {new Date(drone.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}

export default Dashboard;
