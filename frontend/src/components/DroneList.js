import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { 
  Plus, 
  Search, 
  Filter, 
  MoreVertical, 
  Power, 
  PowerOff, 
  Plane, 
  PlaneLanding,
  X
} from 'lucide-react';
import { useDrones } from '../context/DroneContext';
import LocationPicker from './LocationPicker';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMapEvents } from 'react-leaflet';

function WaypointClickCapture({ onAdd }) {
  useMapEvents({
    click(e) {
      onAdd && onAdd(e.latlng);
    },
  });
  return null;
}

function DroneList({ socket }) {
  const {
    drones,
    loading,
    error,
    armDrone,
    disarmDrone,
    takeoffDrone,
    landDrone,
    createDrone,
    deleteDrone,
    gotoDrone,
    emergencyLandAll,
    fetchDrones,
    executeMission,
  } = useDrones();

  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedDrone, setSelectedDrone] = useState(null);
  const [actionBanner, setActionBanner] = useState(null);
  const [pendingByDrone, setPendingByDrone] = useState({});
  const [showGoto, setShowGoto] = useState(false);
  const [gotoAlt, setGotoAlt] = useState(10);
  const [gotoLoc, setGotoLoc] = useState(null);

  const [takeoffAlt, setTakeoffAlt] = useState(10);

  const [detailsLoading, setDetailsLoading] = useState(false);
  const [selectedDetails, setSelectedDetails] = useState(null);

  const [showMission, setShowMission] = useState(false);
  const [missionType, setMissionType] = useState('CUSTOM');
  const [missionDefaultAlt, setMissionDefaultAlt] = useState(10);
  const [missionWaypoints, setMissionWaypoints] = useState([]); // [{lat,lng,alt}]

  const [orbitRadiusM, setOrbitRadiusM] = useState(60);
  const [orbitPoints, setOrbitPoints] = useState(8);
  const [missionTemplateName, setMissionTemplateName] = useState('');
  const [savedTemplates, setSavedTemplates] = useState([]); // [{name, missionType, defaultAlt, waypoints}]
  const [selectedTemplateName, setSelectedTemplateName] = useState('');

  const fleetSummary = useMemo(() => {
    const total = drones.length;
    const armed = drones.filter(d => d.armed).length;
    const flying = drones.filter(d => (d.altitude || 0) > 1).length;
    const lowBattery = drones.filter(d => (d.battery || 100) < 20).length;
    return { total, armed, flying, lowBattery };
  }, [drones]);

  const selectedIsFlying = Boolean(selectedDrone && Number(selectedDrone.altitude || 0) > 1);

  const loadTemplates = () => {
    try {
      const raw = window.localStorage.getItem('lesnar.missions.templates');
      const parsed = raw ? JSON.parse(raw) : [];
      if (Array.isArray(parsed)) setSavedTemplates(parsed);
    } catch {
      setSavedTemplates([]);
    }
  };

  const persistTemplates = (templates) => {
    try {
      window.localStorage.setItem('lesnar.missions.templates', JSON.stringify(templates));
    } catch {
      // ignore
    }
    setSavedTemplates(templates);
  };

  useEffect(() => {
    loadTemplates();
  }, []);

  const waitForAirborne = async (droneId, timeoutMs = 20000) => {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      try {
        const res = await axios.get(`/api/drones/${encodeURIComponent(droneId)}`);
        const st = res.data?.drone;
        if (st && Number(st.altitude || 0) > 1) return true;
      } catch {
        // ignore transient
      }
      // eslint-disable-next-line no-await-in-loop
      await new Promise(r => setTimeout(r, 800));
    }
    return false;
  };

  const ensureAirborneIfNeeded = async (droneId) => {
    if (!selectedDrone) return false;
    if (Number(selectedDrone.altitude || 0) > 1) return true;
    if (!selectedDrone.armed) {
      setActionBanner({ type: 'error', message: 'Drone must be armed before auto takeoff.' });
      return false;
    }
    setActionBanner({ type: 'success', message: `Auto takeoff initiated for ${droneId}` });
    await handleDroneAction('takeoff', droneId, takeoffAlt);
    const ok = await waitForAirborne(droneId);
    if (!ok) {
      setActionBanner({ type: 'error', message: `Timed out waiting for ${droneId} to become airborne` });
    }
    return ok;
  };

  useEffect(() => {
    let cancelled = false;
    const loadDetails = async () => {
      if (!selectedDrone?.drone_id) {
        setSelectedDetails(null);
        return;
      }
      setDetailsLoading(true);
      try {
        const res = await axios.get(`/api/drones/${encodeURIComponent(selectedDrone.drone_id)}`);
        if (cancelled) return;
        if (res.data?.success) {
          setSelectedDetails(res.data);
        } else {
          setSelectedDetails(null);
        }
      } catch {
        if (!cancelled) setSelectedDetails(null);
      } finally {
        if (!cancelled) setDetailsLoading(false);
      }
    };
    loadDetails();
    const id = setInterval(loadDetails, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [selectedDrone?.drone_id]);

  const filteredDrones = drones.filter(drone => {
    const matchesSearch = drone.drone_id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === 'all' || 
      (filterStatus === 'armed' && drone.armed) ||
      (filterStatus === 'flying' && drone.altitude > 1) ||
      (filterStatus === 'landed' && drone.altitude <= 1 && !drone.armed);
    
    return matchesSearch && matchesFilter;
  });

  const handleDroneAction = async (action, droneId, ...args) => {
    try {
      setActionBanner(null);
      setPendingByDrone(prev => ({ ...prev, [droneId]: action }));
      switch (action) {
        case 'arm':
          await armDrone(droneId);
          break;
        case 'disarm':
          await disarmDrone(droneId);
          break;
        case 'takeoff':
          await takeoffDrone(droneId, args[0] || 10);
          break;
        case 'land':
          await landDrone(droneId);
          break;
        case 'delete':
          if (window.confirm(`Are you sure you want to delete ${droneId}?`)) {
            await deleteDrone(droneId);
          }
          break;
        default:
          console.log(`Unknown action: ${action}`);
      }
      setActionBanner({ type: 'success', message: `${action.toUpperCase()} sent to ${droneId}` });
    } catch (error) {
      console.error(`Failed to ${action} drone:`, error);
      setActionBanner({ type: 'error', message: `Failed to ${action} ${droneId}` });
    } finally {
      setPendingByDrone(prev => {
        const next = { ...prev };
        delete next[droneId];
        return next;
      });
    }
  };

  const handleSelectDrone = (drone) => {
    setSelectedDrone(drone);
    setShowGoto(false);
    setShowMission(false);
    setGotoLoc(null);
    setGotoAlt(10);
    setTakeoffAlt(10);
    setMissionType('CUSTOM');
    setMissionDefaultAlt(10);
    setMissionWaypoints([]);
    setOrbitRadiusM(60);
    setOrbitPoints(8);
    setMissionTemplateName('');
    setSelectedTemplateName('');
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Drone Fleet</h1>
            <p className="mt-1 text-sm text-gray-600">
              Manage and control your drone fleet
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={async () => {
                if (!window.confirm('Emergency land ALL drones?')) return;
                try {
                  await emergencyLandAll();
                  setActionBanner({ type: 'success', message: 'Emergency land initiated for all drones' });
                } catch {
                  setActionBanner({ type: 'error', message: 'Emergency land failed' });
                }
              }}
              className="btn-danger"
              title="Emergency land all drones"
            >
              Emergency Land All
            </button>
            <button
              onClick={() => fetchDrones && fetchDrones()}
              className="btn-secondary"
              title="Refresh drone states"
            >
              Refresh
            </button>
            <button
              onClick={() => setShowAddModal(true)}
              className="btn-primary flex items-center"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Drone
            </button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
          <SummaryPill label="Total" value={fleetSummary.total} tone="gray" />
          <SummaryPill label="Armed" value={fleetSummary.armed} tone="yellow" />
          <SummaryPill label="Flying" value={fleetSummary.flying} tone="green" />
          <SummaryPill label="Low Battery" value={fleetSummary.lowBattery} tone="red" />
        </div>
      </div>

      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {actionBanner && (
        <div className={`mb-4 border rounded-lg p-4 ${actionBanner.type === 'success' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div className="flex items-center justify-between">
            <p className={`text-sm ${actionBanner.type === 'success' ? 'text-green-700' : 'text-red-700'}`}>{actionBanner.message}</p>
            <button
              onClick={() => setActionBanner(null)}
              className="p-1 rounded hover:bg-black/5"
              title="Dismiss"
            >
              <X className="h-4 w-4 text-gray-500" />
            </button>
          </div>
        </div>
      )}

      {/* Search and filters */}
      <div className="mb-6 flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search drones..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        
        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-gray-400" />
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Status</option>
            <option value="armed">Armed</option>
            <option value="flying">Flying</option>
            <option value="landed">Landed</option>
          </select>
        </div>
      </div>

      {selectedDrone && (
        <div className="mb-6 bg-white rounded-lg shadow-md border border-gray-200 p-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="text-xs text-gray-500">Selected Drone</div>
              <div className="text-lg font-semibold text-gray-900">{selectedDrone.drone_id}</div>
              <div className="mt-1 text-sm text-gray-600">
                {selectedDrone.mode} • Alt {Number(selectedDrone.altitude || 0).toFixed(1)}m • Bat {Number(selectedDrone.battery || 0).toFixed(0)}%
              </div>
            </div>
            <button
              onClick={() => setSelectedDrone(null)}
              className="p-1 rounded hover:bg-gray-100"
              title="Clear selection"
            >
              <X className="h-4 w-4 text-gray-500" />
            </button>
          </div>

          <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              <div className="text-sm font-medium text-gray-900 mb-2">Details</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <SummaryPill label="Lat" value={Number(selectedDrone.latitude || 0).toFixed(4)} tone="gray" />
                <SummaryPill label="Lng" value={Number(selectedDrone.longitude || 0).toFixed(4)} tone="gray" />
                <SummaryPill label="Speed" value={`${Number(selectedDrone.speed || 0).toFixed(1)} m/s`} tone="gray" />
                <SummaryPill label="Heading" value={`${Number(selectedDrone.heading || 0).toFixed(0)}°`} tone="gray" />
              </div>

              <div className="mt-3 text-xs text-gray-500">
                {detailsLoading ? 'Refreshing details…' : 'Live details'}
                {selectedDrone.timestamp ? ` • Updated ${new Date(selectedDrone.timestamp).toLocaleTimeString()}` : ''}
              </div>

              {selectedDetails?.obstacles && Array.isArray(selectedDetails.obstacles) && (
                <div className="mt-3">
                  <div className="text-sm font-medium text-gray-900 mb-1">Recent Obstacles</div>
                  {selectedDetails.obstacles.length === 0 ? (
                    <div className="text-sm text-gray-600">None</div>
                  ) : (
                    <div className="space-y-2">
                      {selectedDetails.obstacles.slice(-5).reverse().map((o, idx) => (
                        <div key={idx} className="text-sm text-gray-700 bg-gray-50 border rounded px-3 py-2">
                          <span className="font-medium">{o.type || 'obstacle'}</span>
                          {typeof o.distance === 'number' ? ` • ${o.distance.toFixed(1)}m` : ''}
                          {typeof o.bearing === 'number' ? ` • ${o.bearing.toFixed(0)}°` : ''}
                          {o.timestamp ? ` • ${new Date(o.timestamp).toLocaleTimeString()}` : ''}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div>
              <div className="text-sm font-medium text-gray-900 mb-2">Quick Actions</div>
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  {!selectedDrone.armed ? (
                    <button className="btn-secondary" onClick={() => handleDroneAction('arm', selectedDrone.drone_id)}>Arm</button>
                  ) : (
                    <button className="btn-danger" onClick={() => handleDroneAction('disarm', selectedDrone.drone_id)}>Disarm</button>
                  )}
                  {selectedIsFlying ? (
                    <button className="btn-primary" onClick={() => handleDroneAction('land', selectedDrone.drone_id)}>Land</button>
                  ) : (
                    <button
                      className={`btn-success ${!selectedDrone.armed ? 'opacity-50 cursor-not-allowed' : ''}`}
                      disabled={!selectedDrone.armed}
                      onClick={() => handleDroneAction('takeoff', selectedDrone.drone_id, takeoffAlt)}
                    >
                      Takeoff
                    </button>
                  )}
                </div>

                {!selectedIsFlying && (
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Takeoff altitude (m)</label>
                    <input
                      type="number"
                      value={takeoffAlt}
                      onChange={(e) => setTakeoffAlt(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                )}

                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => {
                    setShowGoto(v => !v);
                    setShowMission(false);
                    setGotoLoc(null);
                  }}
                >
                  Go To Location
                </button>

                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => {
                    setShowMission(v => !v);
                    setShowGoto(false);
                  }}
                >
                  Mission Builder
                </button>
              </div>
            </div>
          </div>

          {showGoto && (
            <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <LocationPicker
                  title="Destination"
                  initialLat={Number(selectedDrone.latitude || 40.7128)}
                  initialLng={Number(selectedDrone.longitude || -74.0060)}
                  onConfirm={(loc) => setGotoLoc(loc)}
                />
              </div>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Altitude (m)</label>
                  <input
                    type="number"
                    value={gotoAlt}
                    onChange={(e) => setGotoAlt(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <button
                  type="button"
                  className={`btn-success ${(!gotoLoc || !selectedIsFlying) ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={!gotoLoc || !selectedIsFlying}
                  onClick={async () => {
                    try {
                      if (!gotoLoc) {
                        setActionBanner({ type: 'error', message: 'Please pick a destination and click Confirm.' });
                        return;
                      }
                      if (!selectedIsFlying) {
                        setActionBanner({ type: 'error', message: 'Drone must be airborne for Go To. Takeoff first.' });
                        return;
                      }
                      await gotoDrone(selectedDrone.drone_id, gotoLoc.lat, gotoLoc.lng, gotoAlt);
                      setActionBanner({ type: 'success', message: `GOTO sent to ${selectedDrone.drone_id}` });
                    } catch {
                      setActionBanner({ type: 'error', message: `Failed to GOTO ${selectedDrone.drone_id}` });
                    }
                  }}
                >
                  Send Go To
                </button>

                <button
                  type="button"
                  className={`btn-success ${(!gotoLoc || selectedIsFlying || !selectedDrone?.armed) ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={!gotoLoc || selectedIsFlying || !selectedDrone?.armed}
                  onClick={async () => {
                    try {
                      if (!gotoLoc) {
                        setActionBanner({ type: 'error', message: 'Please pick a destination and click Confirm.' });
                        return;
                      }
                      const ok = await ensureAirborneIfNeeded(selectedDrone.drone_id);
                      if (!ok) return;
                      await gotoDrone(selectedDrone.drone_id, gotoLoc.lat, gotoLoc.lng, gotoAlt);
                      setActionBanner({ type: 'success', message: `Takeoff + GOTO sent to ${selectedDrone.drone_id}` });
                    } catch {
                      setActionBanner({ type: 'error', message: `Failed Takeoff + GOTO for ${selectedDrone.drone_id}` });
                    }
                  }}
                >
                  Takeoff + Go To
                </button>
                <div className="text-xs text-gray-500">
                  Destination: {gotoLoc ? (gotoLoc.label ? gotoLoc.label : `${gotoLoc.lat.toFixed(6)}, ${gotoLoc.lng.toFixed(6)}`) : 'Not confirmed'}
                </div>
                {!selectedIsFlying && (
                  <div className="text-xs text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-2">
                    Go To requires the drone to be airborne. Use Takeoff + Go To (requires Armed).
                  </div>
                )}
              </div>
            </div>
          )}

          {showMission && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-medium text-gray-900">Mission Builder</div>
                <button
                  type="button"
                  className="text-sm text-blue-700 hover:text-blue-800"
                  onClick={() => setMissionWaypoints([])}
                >
                  Clear waypoints
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="border rounded-md overflow-hidden">
                  <MapContainer
                    center={[Number(selectedDrone.latitude || 40.7128), Number(selectedDrone.longitude || -74.0060)]}
                    zoom={13}
                    scrollWheelZoom={true}
                    zoomControl={true}
                    className="w-full"
                    style={{ height: 280 }}
                  >
                    <TileLayer
                      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    <WaypointClickCapture
                      onAdd={(ll) => setMissionWaypoints(prev => ([...prev, { lat: ll.lat, lng: ll.lng, alt: missionDefaultAlt }]))}
                    />

                    {missionWaypoints.length > 0 && (
                      <Polyline
                        positions={missionWaypoints.map(w => [w.lat, w.lng])}
                        pathOptions={{ color: '#2563eb', weight: 3 }}
                      />
                    )}

                    {missionWaypoints.map((w, idx) => (
                      <Marker key={`${w.lat}-${w.lng}-${idx}`} position={[w.lat, w.lng]}>
                        <Popup>
                          <div className="text-sm">
                            <div className="font-medium">Waypoint {idx + 1}</div>
                            <div>{w.lat.toFixed(6)}, {w.lng.toFixed(6)}</div>
                            <div>Alt: {Number(w.alt || 0).toFixed(1)}m</div>
                          </div>
                        </Popup>
                      </Marker>
                    ))}
                  </MapContainer>
                </div>

                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Mission Type</label>
                      <input
                        type="text"
                        value={missionType}
                        onChange={(e) => setMissionType(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="CUSTOM"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Default Alt (m)</label>
                      <input
                        type="number"
                        value={missionDefaultAlt}
                        onChange={(e) => {
                          const nextAlt = Number(e.target.value);
                          setMissionDefaultAlt(nextAlt);
                        }}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </div>

                  <div className="text-sm text-gray-700">
                    Click the map to add waypoints. Dragging isn’t enabled yet—use remove/edit below.
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Orbit radius (m)</label>
                      <input
                        type="number"
                        value={orbitRadiusM}
                        onChange={(e) => setOrbitRadiusM(Number(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Orbit points</label>
                      <input
                        type="number"
                        value={orbitPoints}
                        onChange={(e) => setOrbitPoints(Number(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </div>

                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() => {
                      const centerLat = Number(selectedDrone.latitude || 40.7128);
                      const centerLng = Number(selectedDrone.longitude || -74.0060);
                      const pts = Math.max(3, Math.min(48, Number(orbitPoints || 8)));
                      const r = Math.max(5, Math.min(2000, Number(orbitRadiusM || 60)));
                      // Approx meters per degree
                      const latFactor = 1 / 111000;
                      const lonFactor = 1 / (111000 * Math.cos((centerLat * Math.PI) / 180));
                      const gen = [];
                      for (let i = 0; i < pts; i += 1) {
                        const a = (2 * Math.PI * i) / pts;
                        const dLat = Math.sin(a) * r * latFactor;
                        const dLng = Math.cos(a) * r * lonFactor;
                        gen.push({ lat: centerLat + dLat, lng: centerLng + dLng, alt: missionDefaultAlt });
                      }
                      setMissionWaypoints(gen);
                      setMissionType('ORBIT');
                    }}
                  >
                    Generate Orbit Mission
                  </button>

                  <div className="border rounded-md overflow-auto max-h-48">
                    {missionWaypoints.length === 0 ? (
                      <div className="px-3 py-3 text-sm text-gray-500">No waypoints yet.</div>
                    ) : (
                      missionWaypoints.map((w, idx) => (
                        <div key={idx} className="flex items-center justify-between px-3 py-2 border-t first:border-t-0">
                          <div className="text-sm">
                            <div className="font-medium">#{idx + 1}</div>
                            <div className="text-xs text-gray-500">{w.lat.toFixed(6)}, {w.lng.toFixed(6)}</div>
                          </div>
                          <div className="flex items-center gap-2">
                            <input
                              type="number"
                              value={Number(w.alt || missionDefaultAlt)}
                              onChange={(e) => {
                                const nextAlt = Number(e.target.value);
                                setMissionWaypoints(prev => prev.map((p, i) => i === idx ? ({ ...p, alt: nextAlt }) : p));
                              }}
                              className="w-24 px-2 py-1 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              title="Waypoint altitude"
                            />
                            <button
                              type="button"
                              className={`text-sm text-gray-700 hover:text-gray-900 ${idx === 0 ? 'opacity-40 cursor-not-allowed' : ''}`}
                              disabled={idx === 0}
                              onClick={() => {
                                setMissionWaypoints(prev => {
                                  const next = [...prev];
                                  const tmp = next[idx - 1];
                                  next[idx - 1] = next[idx];
                                  next[idx] = tmp;
                                  return next;
                                });
                              }}
                            >
                              Up
                            </button>
                            <button
                              type="button"
                              className={`text-sm text-gray-700 hover:text-gray-900 ${idx === missionWaypoints.length - 1 ? 'opacity-40 cursor-not-allowed' : ''}`}
                              disabled={idx === missionWaypoints.length - 1}
                              onClick={() => {
                                setMissionWaypoints(prev => {
                                  const next = [...prev];
                                  const tmp = next[idx + 1];
                                  next[idx + 1] = next[idx];
                                  next[idx] = tmp;
                                  return next;
                                });
                              }}
                            >
                              Down
                            </button>
                            <button
                              type="button"
                              className="text-sm text-red-700 hover:text-red-800"
                              onClick={() => setMissionWaypoints(prev => prev.filter((_, i) => i !== idx))}
                            >
                              Remove
                            </button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Template name</label>
                      <input
                        type="text"
                        value={missionTemplateName}
                        onChange={(e) => setMissionTemplateName(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="e.g., Perimeter Sweep"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Load template</label>
                      <select
                        value={selectedTemplateName}
                        onChange={(e) => {
                          const name = e.target.value;
                          setSelectedTemplateName(name);
                          const t = savedTemplates.find(x => x.name === name);
                          if (t) {
                            setMissionType(t.missionType || 'CUSTOM');
                            setMissionDefaultAlt(Number(t.defaultAlt || 10));
                            setMissionWaypoints(Array.isArray(t.waypoints) ? t.waypoints : []);
                          }
                        }}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      >
                        <option value="">Select…</option>
                        {savedTemplates.map(t => (
                          <option key={t.name} value={t.name}>{t.name}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      className={`btn-secondary ${(!missionTemplateName || missionWaypoints.length === 0) ? 'opacity-50 cursor-not-allowed' : ''}`}
                      disabled={!missionTemplateName || missionWaypoints.length === 0}
                      onClick={() => {
                        const name = missionTemplateName.trim();
                        if (!name) return;
                        const next = savedTemplates.filter(t => t.name !== name);
                        next.unshift({ name, missionType, defaultAlt: missionDefaultAlt, waypoints: missionWaypoints });
                        persistTemplates(next.slice(0, 20));
                        setSelectedTemplateName(name);
                        setActionBanner({ type: 'success', message: `Saved template: ${name}` });
                      }}
                    >
                      Save Template
                    </button>
                    <button
                      type="button"
                      className={`btn-danger ${(!selectedTemplateName) ? 'opacity-50 cursor-not-allowed' : ''}`}
                      disabled={!selectedTemplateName}
                      onClick={() => {
                        const name = selectedTemplateName;
                        const next = savedTemplates.filter(t => t.name !== name);
                        persistTemplates(next);
                        setSelectedTemplateName('');
                        setActionBanner({ type: 'success', message: `Deleted template: ${name}` });
                      }}
                    >
                      Delete Template
                    </button>
                  </div>

                  <button
                    type="button"
                    className={`btn-success ${(missionWaypoints.length === 0 || !selectedIsFlying) ? 'opacity-50 cursor-not-allowed' : ''}`}
                    disabled={missionWaypoints.length === 0 || !selectedIsFlying}
                    onClick={async () => {
                      try {
                        if (!selectedIsFlying) {
                          setActionBanner({ type: 'error', message: 'Drone must be airborne to start a mission. Takeoff first.' });
                          return;
                        }
                        if (missionWaypoints.length === 0) {
                          setActionBanner({ type: 'error', message: 'Add at least one waypoint.' });
                          return;
                        }
                        const payload = missionWaypoints.map(w => [w.lat, w.lng, Number(w.alt || missionDefaultAlt)]);
                        await executeMission(selectedDrone.drone_id, payload, missionType || 'CUSTOM');
                        setActionBanner({ type: 'success', message: `Mission sent to ${selectedDrone.drone_id}` });
                      } catch {
                        setActionBanner({ type: 'error', message: `Failed to start mission for ${selectedDrone.drone_id}` });
                      }
                    }}
                  >
                    Send Mission
                  </button>

                  <button
                    type="button"
                    className={`btn-success ${(missionWaypoints.length === 0 || selectedIsFlying || !selectedDrone?.armed) ? 'opacity-50 cursor-not-allowed' : ''}`}
                    disabled={missionWaypoints.length === 0 || selectedIsFlying || !selectedDrone?.armed}
                    onClick={async () => {
                      try {
                        if (missionWaypoints.length === 0) {
                          setActionBanner({ type: 'error', message: 'Add at least one waypoint.' });
                          return;
                        }
                        const ok = await ensureAirborneIfNeeded(selectedDrone.drone_id);
                        if (!ok) return;
                        const payload = missionWaypoints.map(w => [w.lat, w.lng, Number(w.alt || missionDefaultAlt)]);
                        await executeMission(selectedDrone.drone_id, payload, missionType || 'CUSTOM');
                        setActionBanner({ type: 'success', message: `Takeoff + Mission sent to ${selectedDrone.drone_id}` });
                      } catch {
                        setActionBanner({ type: 'error', message: `Failed Takeoff + Mission for ${selectedDrone.drone_id}` });
                      }
                    }}
                  >
                    Takeoff + Mission
                  </button>

                  {!selectedIsFlying && (
                    <div className="text-xs text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-2">
                      Missions require the drone to be airborne. Use Takeoff + Mission (requires Armed).
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Drone grid */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="spinner"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredDrones.map((drone) => (
            <DroneCard
              key={drone.drone_id}
              drone={drone}
              onAction={handleDroneAction}
              onSelect={handleSelectDrone}
              pendingAction={pendingByDrone[drone.drone_id]}
            />
          ))}
          
          {filteredDrones.length === 0 && (
            <div className="col-span-full text-center py-12">
              <div className="text-gray-400 mb-4">
                {searchTerm || filterStatus !== 'all' ? 
                  'No drones match your filters' : 
                  'No drones in your fleet'
                }
              </div>
              {!searchTerm && filterStatus === 'all' && (
                <button
                  onClick={() => setShowAddModal(true)}
                  className="btn-primary"
                >
                  Add your first drone
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Add Drone Modal */}
      {showAddModal && (
        <AddDroneModal
          onClose={() => setShowAddModal(false)}
          onAdd={createDrone}
        />
      )}
    </div>
  );
}

function SummaryPill({ label, value, tone }) {
  const toneClasses = {
    gray: 'bg-gray-100 text-gray-800',
    green: 'bg-green-100 text-green-800',
    yellow: 'bg-yellow-100 text-yellow-800',
    red: 'bg-red-100 text-red-800',
  };
  return (
    <div className={`rounded-lg px-4 py-3 ${toneClasses[tone] || toneClasses.gray}`}>
      <div className="text-xs font-medium opacity-80">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}

function DroneCard({ drone, onAction, onSelect, pendingAction }) {
  const [showMenu, setShowMenu] = useState(false);
  
  const getStatusColor = () => {
    if (drone.mode === 'EMERGENCY') return 'red';
    if (drone.altitude > 1 && drone.armed) return 'green';
    if (drone.armed) return 'yellow';
    return 'gray';
  };

  const getBatteryColor = () => {
    if (drone.battery >= 60) return 'green';
    if (drone.battery >= 30) return 'yellow';
    return 'red';
  };

  const statusColor = getStatusColor();
  const batteryColor = getBatteryColor();

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          {drone.drone_id}
        </h3>
        
        <div className="flex items-center space-x-2">
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
          
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-1 rounded-md hover:bg-gray-100"
            >
              <MoreVertical className="h-4 w-4 text-gray-400" />
            </button>
            
            {showMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-10 border">
                <div className="py-1">
                  <button
                    onClick={() => {
                      onSelect(drone);
                      setShowMenu(false);
                    }}
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    View Details
                  </button>
                  <button
                    onClick={() => {
                      onAction('delete', drone.drone_id);
                      setShowMenu(false);
                    }}
                    className="block w-full text-left px-4 py-2 text-sm text-red-700 hover:bg-red-50"
                  >
                    Delete Drone
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-900">
            {Number(drone.altitude || 0).toFixed(1)}m
          </div>
          <div className="text-xs text-gray-500">Altitude</div>
        </div>
        
        <div className="text-center">
          <div className={`text-lg font-semibold ${
            batteryColor === 'green' ? 'text-green-600' :
            batteryColor === 'yellow' ? 'text-yellow-600' :
            'text-red-600'
          }`}>
            {Number(drone.battery || 0).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500">Battery</div>
        </div>
        
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-900">
            {Number(drone.speed || 0).toFixed(1)}
          </div>
          <div className="text-xs text-gray-500">m/s</div>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-2 gap-2">
        {!drone.armed ? (
          <button
            onClick={() => onAction('arm', drone.drone_id)}
            className={`btn-secondary flex items-center justify-center text-sm ${pendingAction ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={Boolean(pendingAction)}
          >
            <Power className="h-4 w-4 mr-1" />
            {pendingAction === 'arm' ? 'Arming...' : 'Arm'}
          </button>
        ) : (
          <button
            onClick={() => onAction('disarm', drone.drone_id)}
            className={`btn-danger flex items-center justify-center text-sm ${pendingAction ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={Boolean(pendingAction)}
          >
            <PowerOff className="h-4 w-4 mr-1" />
            {pendingAction === 'disarm' ? 'Disarming...' : 'Disarm'}
          </button>
        )}
        
        {drone.altitude <= 1 ? (
          <button
            onClick={() => onAction('takeoff', drone.drone_id, 10)}
            className={`btn-success flex items-center justify-center text-sm ${(!drone.armed || pendingAction) ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={!drone.armed || Boolean(pendingAction)}
          >
            <Plane className="h-4 w-4 mr-1" />
            {pendingAction === 'takeoff' ? 'Taking off...' : 'Takeoff'}
          </button>
        ) : (
          <button
            onClick={() => onAction('land', drone.drone_id)}
            className={`btn-primary flex items-center justify-center text-sm ${pendingAction ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={Boolean(pendingAction)}
          >
            <PlaneLanding className="h-4 w-4 mr-1" />
            {pendingAction === 'land' ? 'Landing...' : 'Land'}
          </button>
        )}
      </div>

      {/* Last update */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          Last update: {drone.timestamp ? new Date(drone.timestamp).toLocaleTimeString() : '—'}
        </p>
      </div>
    </div>
  );
}

function AddDroneModal({ onClose, onAdd }) {
  const [formData, setFormData] = useState({
    drone_id: '',
    latitude: 40.7128,
    longitude: -74.0060,
    altitude: 0
  });
  const [confirmedLoc, setConfirmedLoc] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setSubmitError(null);
      if (!confirmedLoc) {
        setSubmitError('Please pick a start location and click Confirm.');
        return;
      }
      const lat = Number(confirmedLoc.lat);
      const lng = Number(confirmedLoc.lng);
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
        setSubmitError('Invalid confirmed location. Please confirm again.');
        return;
      }

      await onAdd({
        drone_id: formData.drone_id,
        position: [lat, lng, formData.altitude]
      });
      onClose();
    } catch (error) {
      console.error('Failed to add drone:', error);
      setSubmitError('Failed to add drone');
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h2 className="text-lg font-semibold mb-4">Add New Drone</h2>

        {submitError && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-3">
            <p className="text-sm text-red-700">{submitError}</p>
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Drone ID
            </label>
            <input
              type="text"
              value={formData.drone_id}
              onChange={(e) => setFormData({...formData, drone_id: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="LESNAR-001"
              required
            />
          </div>
          
          <LocationPicker
            title="Start Location"
            initialLat={formData.latitude}
            initialLng={formData.longitude}
            onConfirm={(loc) => {
              setConfirmedLoc(loc);
              setFormData(prev => ({ ...prev, latitude: loc.lat, longitude: loc.lng }));
            }}
          />

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Start Altitude (m)
            </label>
            <input
              type="number"
              step="0.1"
              value={formData.altitude}
              onChange={(e) => setFormData({ ...formData, altitude: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              className={`btn-primary ${!confirmedLoc ? 'opacity-50 cursor-not-allowed' : ''}`}
              disabled={!confirmedLoc}
            >
              Add Drone
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default DroneList;
