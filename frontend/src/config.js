export const BACKEND_URL = String(
  process.env.REACT_APP_BACKEND_URL ||
    process.env.REACT_APP_API_BASE_URL ||
    'http://localhost:5000'
).replace(/\/$/, '');

export const API_KEY = (process.env.REACT_APP_API_KEY || '').trim();
