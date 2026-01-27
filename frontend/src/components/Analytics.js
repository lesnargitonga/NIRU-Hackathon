import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { BarChart3, TrendingUp, Activity, Clock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Sample data
const sampleFlightTimeData = [
  { time: '00:00', flights: 0 },
  { time: '04:00', flights: 2 },
  { time: '08:00', flights: 8 },
  { time: '12:00', flights: 15 },
  { time: '16:00', flights: 12 },
  { time: '20:00', flights: 6 },
  { time: '24:00', flights: 1 },
];

function Analytics({ socket }) {
  const [segLog, setSegLog] = useState({ file: null, rows: [] });
  const [flightData, setFlightData] = useState(sampleFlightTimeData);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadSegmentationLog = async () => {
      setError(null);
      try {
        const res = await axios.get('/api/logs/segmentation/latest');
        if (res.data?.success && Array.isArray(res.data.rows)) {
          setSegLog({ file: res.data.file, rows: res.data.rows });
          // Build a simple time series if timestamps exist
          const countsByHour = {};
          res.data.rows.forEach(row => {
            const t = row.timestamp || row.time || null;
            if (t) {
              try {
                const d = new Date(t);
                const h = d.getHours().toString().padStart(2, '0');
                countsByHour[h] = (countsByHour[h] || 0) + 1;
              } catch {}
            }
          });
          const chart = Object.keys(countsByHour).sort().map(h => ({ time: `${h}:00`, flights: countsByHour[h] }));
          if (chart.length > 0) setFlightData(chart);
        } else {
          setError(res.data?.error || 'No segmentation logs found');
        }
      } catch (e) {
        setError('Failed to load latest segmentation log');
      }
    };
    loadSegmentationLog();
  }, []);
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
        <p className="mt-1 text-sm text-gray-600">
          Fleet performance metrics and insights
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Total Flight Time"
          value="142.5 hrs"
          change="+12%"
          icon={Clock}
          trend="up"
        />
        <MetricCard
          title="Missions Completed"
          value="87"
          change="+8%"
          icon={BarChart3}
          trend="up"
        />
        <MetricCard
          title="Average Battery Life"
          value="28.3 min"
          change="-3%"
          icon={Activity}
          trend="down"
        />
        <MetricCard
          title="System Uptime"
          value="99.2%"
          change="+0.5%"
          icon={TrendingUp}
          trend="up"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Flight Activity Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold mb-4">Flight Activity (24hrs)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={flightData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="flights" 
                stroke="#3B82F6" 
                strokeWidth={2}
                dot={{ fill: '#3B82F6' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Fleet Health */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold mb-4">Fleet Health Status</h2>
          <div className="space-y-4">
            <HealthItem label="Battery Health" value={92} color="green" />
            <HealthItem label="Motor Performance" value={88} color="green" />
            <HealthItem label="GPS Accuracy" value={95} color="green" />
            <HealthItem label="Communication Link" value={76} color="yellow" />
            <HealthItem label="Sensor Calibration" value={84} color="green" />
          </div>
        </div>
      </div>

      {/* Segmentation Logs Summary */}
      <div className="bg-white rounded-lg shadow-md p-6 mt-6">
        <h3 className="text-lg font-semibold mb-2">Latest Segmentation Log</h3>
        {error ? (
          <p className="text-sm text-red-600">{error}</p>
        ) : segLog.file ? (
          <div>
            <p className="text-sm text-gray-600 mb-2">File: {segLog.file}</p>
            <p className="text-sm text-gray-600 mb-4">Entries: {segLog.rows.length}</p>
            <div className="overflow-auto max-h-64 border rounded">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    {Object.keys(segLog.rows[0] || {}).slice(0,6).map(k => (
                      <th key={k} className="text-left px-3 py-2 font-medium text-gray-700">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {segLog.rows.slice(0,20).map((row, idx) => (
                    <tr key={idx} className="border-t">
                      {Object.keys(segLog.rows[0] || {}).slice(0,6).map(k => (
                        <td key={k} className="px-3 py-2 text-gray-800">{String(row[k])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-600">No log data available</p>
        )}
      </div>

      {/* Additional Analytics Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Top Performing Drones</h3>
          <div className="space-y-3">
            <DroneRanking rank="1" id="LESNAR-001" score="98%" />
            <DroneRanking rank="2" id="LESNAR-003" score="94%" />
            <DroneRanking rank="3" id="LESNAR-002" score="91%" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Mission Success Rate</h3>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">94.2%</div>
            <p className="text-sm text-gray-600">This month</p>
            <div className="mt-4 bg-gray-200 rounded-full h-2">
              <div className="bg-green-600 h-2 rounded-full" style={{width: '94.2%'}}></div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Alerts & Warnings</h3>
          <div className="space-y-2">
            <AlertItem type="warning" message="LESNAR-002 low battery" time="2 min ago" />
            <AlertItem type="info" message="Mission completed successfully" time="15 min ago" />
            <AlertItem type="success" message="All systems operational" time="1 hr ago" />
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, change, icon: Icon, trend }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
        <Icon className="h-8 w-8 text-blue-600" />
      </div>
      <div className="mt-4">
        <span className={`text-sm font-medium ${
          trend === 'up' ? 'text-green-600' : 'text-red-600'
        }`}>
          {change}
        </span>
        <span className="text-sm text-gray-600 ml-1">from last period</span>
      </div>
    </div>
  );
}

function HealthItem({ label, value, color }) {
  const colorClasses = {
    green: 'bg-green-600',
    yellow: 'bg-yellow-600',
    red: 'bg-red-600'
  };

  return (
    <div className="flex items-center justify-between">
      <span className="text-sm font-medium text-gray-700">{label}</span>
      <div className="flex items-center space-x-2">
        <div className="w-24 bg-gray-200 rounded-full h-2">
          <div 
            className={`h-2 rounded-full ${colorClasses[color]}`}
            style={{ width: `${value}%` }}
          ></div>
        </div>
        <span className="text-sm font-medium text-gray-900 w-10">{value}%</span>
      </div>
    </div>
  );
}

function DroneRanking({ rank, id, score }) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="flex items-center justify-center w-6 h-6 bg-blue-100 text-blue-800 rounded-full text-xs font-bold">
          {rank}
        </div>
        <span className="text-sm font-medium">{id}</span>
      </div>
      <span className="text-sm font-bold text-green-600">{score}</span>
    </div>
  );
}

function AlertItem({ type, message, time }) {
  const typeClasses = {
    warning: 'bg-yellow-100 text-yellow-800',
    info: 'bg-blue-100 text-blue-800',
    success: 'bg-green-100 text-green-800',
    error: 'bg-red-100 text-red-800'
  };

  return (
    <div className="flex items-start space-x-2">
      <div className={`px-2 py-1 rounded text-xs font-medium ${typeClasses[type]}`}>
        {type.toUpperCase()}
      </div>
      <div className="flex-1">
        <p className="text-sm text-gray-900">{message}</p>
        <p className="text-xs text-gray-500">{time}</p>
      </div>
    </div>
  );
}

export default Analytics;
