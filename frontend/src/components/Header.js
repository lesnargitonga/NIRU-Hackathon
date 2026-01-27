import React from 'react';
import { Link } from 'react-router-dom';
import { Menu, Wifi, WifiOff, Bell, Map as MapIcon } from 'lucide-react';

function Header({ onMenuClick, connected }) {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="flex items-center justify-between px-4 py-3">
        {/* Left side */}
        <div className="flex items-center">
          <button
            onClick={onMenuClick}
            className="p-2 rounded-md hover:bg-gray-100 lg:hidden"
          >
            <Menu className="h-5 w-5 text-gray-600" />
          </button>
          
          <div className="ml-4 lg:ml-0">
            <h1 className="text-xl font-bold text-gray-900">
              Lesnar AI
            </h1>
            <p className="text-sm text-gray-500">
              Drone Control Dashboard
            </p>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* Connection status */}
          <div className="flex items-center space-x-2">
            {connected ? (
              <>
                <Wifi className="h-5 w-5 text-green-500" />
                <span className="text-sm text-green-600 font-medium">
                  Connected
                </span>
              </>
            ) : (
              <>
                <WifiOff className="h-5 w-5 text-red-500" />
                <span className="text-sm text-red-600 font-medium">
                  Disconnected
                </span>
              </>
            )}
          </div>

          {/* Notifications */}
          <button className="p-2 rounded-md hover:bg-gray-100 relative">
            <Bell className="h-5 w-5 text-gray-600" />
            <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-500 rounded-full"></span>
          </button>

          {/* Map quick access */}
          <button className="p-2 rounded-md hover:bg-gray-100">
            <MapIcon className="h-5 w-5 text-gray-600" />
          </button>

          {/* Live Map quick access - new button */}
          <Link
            to="/map"
            className="hidden md:flex items-center px-3 py-1.5 rounded-md bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200"
            title="Open Live Map"
          >
            <MapIcon className="h-4 w-4 mr-2" /> Live Map
          </Link>

          {/* User menu */}
          <div className="flex items-center">
            <div className="h-8 w-8 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full flex items-center justify-center">
              <span className="text-sm font-medium text-white">LA</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
