import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { Navigation, MapPin, Clock, Pause, Play, Square } from 'lucide-react';
import { MapContainer, Marker, Polyline, Popup, TileLayer, useMapEvents } from 'react-leaflet';
import { useDrones } from '../context/DroneContext';

function WaypointClickCapture({ onAdd }) {
  useMapEvents({
    click(e) {
      onAdd && onAdd(e.latlng);
    },
  });
  return null;
}

function formatRemaining(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return '—';
  const s = Math.max(0, Number(seconds));
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
}

function MissionControl({ socket }) {
  const { drones, fetchDrones, executeMission, takeoffDrone } = useDrones();

  const [selectedDroneId, setSelectedDroneId] = useState('');
  const [actionBanner, setActionBanner] = useState(null);
  const [pending, setPending] = useState(false);

  const [missionType, setMissionType] = useState('CUSTOM');
  const [missionDefaultAlt, setMissionDefaultAlt] = useState(10);
  const [missionWaypoints, setMissionWaypoints] = useState([]); // [{lat,lng,alt}]

  const [orbitRadiusM, setOrbitRadiusM] = useState(60);
  const [orbitPoints, setOrbitPoints] = useState(8);

  const [showAdvanced, setShowAdvanced] = useState(false);

  const [missionTemplateName, setMissionTemplateName] = useState('');
  const [savedTemplates, setSavedTemplates] = useState([]);
  const [selectedTemplateName, setSelectedTemplateName] = useState('');

  const [activeMissions, setActiveMissions] = useState([]);
  const [missionsLoading, setMissionsLoading] = useState(false);

  const selectedDrone = useMemo(() => drones.find(d => d.drone_id === selectedDroneId) || null, [drones, selectedDroneId]);
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

  useEffect(() => {
    if (!selectedDroneId && drones.length > 0) setSelectedDroneId(drones[0].drone_id);
  }, [drones, selectedDroneId]);

  useEffect(() => {
    fetchDrones && fetchDrones();
  }, [fetchDrones]);

  const refreshActiveMissions = async () => {
    setMissionsLoading(true);
    try {
      const res = await axios.get('/api/missions/active');
      if (res.data?.success) setActiveMissions(res.data.missions || []);
      else setActiveMissions([]);
    } catch {
      setActiveMissions([]);
    } finally {
      setMissionsLoading(false);
    }
  };

  useEffect(() => {
    refreshActiveMissions();
    const id = setInterval(refreshActiveMissions, 5000);
    return () => clearInterval(id);
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
    await takeoffDrone(droneId, missionDefaultAlt);
    const ok = await waitForAirborne(droneId);
    if (!ok) setActionBanner({ type: 'error', message: `Timed out waiting for ${droneId} to become airborne` });
    return ok;
  };

  const mapCenter = useMemo(() => {
    const lat = Number(selectedDrone?.latitude ?? 40.7128);
    const lng = Number(selectedDrone?.longitude ?? -74.0060);
    return [lat, lng];
  }, [selectedDrone]);

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Mission Control</h1>
        <p className="mt-1 text-sm text-gray-600">Plan, execute, and monitor drone missions</p>
      </div>

      {actionBanner && (
        <div className={`mb-4 rounded-md border px-4 py-3 text-sm ${actionBanner.type === 'error' ? 'bg-red-50 border-red-200 text-red-800' : 'bg-green-50 border-green-200 text-green-800'}`}>
          {actionBanner.message}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Mission Planning */}
        <div className="lg:col-span-2 card">
          <div className="flex items-center justify-between mb-4 gap-3">
            <h2 className="text-lg font-semibold flex items-center">
              <Navigation className="h-5 w-5 mr-2" />
              Mission Planning
            </h2>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-700">Drone</label>
              <select
                value={selectedDroneId}
                onChange={(e) => {
                  setSelectedDroneId(e.target.value);
                  setMissionWaypoints([]);
                  setMissionType('CUSTOM');
                  setMissionDefaultAlt(10);
                  setOrbitRadiusM(60);
                  setOrbitPoints(8);
                  setMissionTemplateName('');
                  setSelectedTemplateName('');
                }}
                className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {drones.map(d => (
                  <option key={d.drone_id} value={d.drone_id}>{d.drone_id}</option>
                ))}
              </select>
            </div>
          </div>

          {!selectedDrone ? (
            <div className="text-center py-12 text-gray-500">
              <p>No drones available. Create a drone in Drone Fleet.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="border rounded-md overflow-hidden">
                <MapContainer
                  center={mapCenter}
                  zoom={13}
                  scrollWheelZoom={true}
                  zoomControl={true}
                  className="w-full"
                  style={{ height: 360 }}
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
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-gray-900">Waypoints</div>
                  <button type="button" className="text-sm text-blue-700 hover:text-blue-800" onClick={() => setMissionWaypoints([])}>
                    Clear
                  </button>
                </div>

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
                      onChange={(e) => setMissionDefaultAlt(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <div className="text-sm text-gray-700">Click the map to add waypoints. Use edit/reorder below.</div>

                <div className="flex items-center justify-between pt-1">
                  <div className="text-xs text-gray-500">Primary flow: add waypoints → Takeoff + Mission</div>
                  <button
                    type="button"
                    className="text-sm text-blue-700 hover:text-blue-800"
                    onClick={() => setShowAdvanced(v => !v)}
                  >
                    {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
                  </button>
                </div>

                {showAdvanced && (
                  <>
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
                  </>
                )}

                <div className="border rounded-md overflow-auto max-h-52">
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
                          <button type="button" className="text-sm text-red-700 hover:text-red-800" onClick={() => setMissionWaypoints(prev => prev.filter((_, i) => i !== idx))}>
                            Remove
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>

                {showAdvanced && (
                  <>
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
                  </>
                )}

                <button
                  type="button"
                  className={`btn-success ${(missionWaypoints.length === 0 || !selectedIsFlying || pending) ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={missionWaypoints.length === 0 || !selectedIsFlying || pending}
                  onClick={async () => {
                    try {
                      setPending(true);
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
                      refreshActiveMissions();
                    } catch {
                      setActionBanner({ type: 'error', message: `Failed to start mission for ${selectedDrone.drone_id}` });
                    } finally {
                      setPending(false);
                    }
                  }}
                >
                  Send Mission
                </button>

                <button
                  type="button"
                  className={`btn-success ${(missionWaypoints.length === 0 || selectedIsFlying || !selectedDrone?.armed || pending) ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={missionWaypoints.length === 0 || selectedIsFlying || !selectedDrone?.armed || pending}
                  onClick={async () => {
                    try {
                      setPending(true);
                      if (missionWaypoints.length === 0) {
                        setActionBanner({ type: 'error', message: 'Add at least one waypoint.' });
                        return;
                      }
                      const ok = await ensureAirborneIfNeeded(selectedDrone.drone_id);
                      if (!ok) return;
                      const payload = missionWaypoints.map(w => [w.lat, w.lng, Number(w.alt || missionDefaultAlt)]);
                      await executeMission(selectedDrone.drone_id, payload, missionType || 'CUSTOM');
                      setActionBanner({ type: 'success', message: `Takeoff + Mission sent to ${selectedDrone.drone_id}` });
                      refreshActiveMissions();
                    } catch {
                      setActionBanner({ type: 'error', message: `Failed Takeoff + Mission for ${selectedDrone.drone_id}` });
                    } finally {
                      setPending(false);
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
          )}
        </div>

        {/* Active Missions */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Active Missions</h2>
            <button type="button" className="btn-secondary" onClick={() => refreshActiveMissions()} disabled={missionsLoading}>
              Refresh
            </button>
          </div>

          {missionsLoading ? (
            <div className="text-sm text-gray-500">Loading…</div>
          ) : (
            <div className="space-y-3">
              {activeMissions.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <Play className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No active missions</p>
                </div>
              ) : (
                activeMissions.map((m) => (
                  <div key={m.drone_id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{m.drone_id}</h3>
                      <span className={`px-2 py-1 text-xs rounded ${m.status === 'PAUSED' ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}`}>
                        {m.status === 'PAUSED' ? 'Paused' : 'Active'}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 space-y-1">
                      <div className="flex items-center">
                        <MapPin className="h-4 w-4 mr-1" />
                        <span>
                          {m.mission_type || 'CUSTOM'}
                          {m.total_waypoints != null ? ` • ${m.total_waypoints} waypoints` : ''}
                          {m.current_waypoint_index != null && m.total_waypoints != null ? ` • wp ${m.current_waypoint_index}/${m.total_waypoints}` : ''}
                        </span>
                      </div>
                      <div className="flex items-center">
                        <Clock className="h-4 w-4 mr-1" />
                        <span>{formatRemaining(m.estimated_remaining_s)} remaining</span>
                      </div>
                    </div>
                    <div className="flex space-x-2 mt-3">
                      <button
                        type="button"
                        className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-3 rounded-md transition-colors flex items-center justify-center"
                        onClick={async () => {
                          try {
                            const endpoint = m.status === 'PAUSED' ? 'resume' : 'pause';
                            await axios.post(`/api/drones/${encodeURIComponent(m.drone_id)}/mission/${endpoint}`);
                            await refreshActiveMissions();
                          } catch {
                            setActionBanner({ type: 'error', message: `Failed to ${m.status === 'PAUSED' ? 'resume' : 'pause'} mission for ${m.drone_id}` });
                          }
                        }}
                      >
                        {m.status === 'PAUSED' ? (
                          <>
                            <Play className="h-4 w-4 mr-1" />
                            Resume
                          </>
                        ) : (
                          <>
                            <Pause className="h-4 w-4 mr-1" />
                            Pause
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        className="flex-1 btn-danger flex items-center justify-center"
                        onClick={async () => {
                          if (!window.confirm(`Stop mission for ${m.drone_id}?`)) return;
                          try {
                            await axios.post(`/api/drones/${encodeURIComponent(m.drone_id)}/mission/stop`);
                            await refreshActiveMissions();
                          } catch {
                            setActionBanner({ type: 'error', message: `Failed to stop mission for ${m.drone_id}` });
                          }
                        }}
                      >
                        <Square className="h-4 w-4 mr-1" />
                        Stop
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default MissionControl;
