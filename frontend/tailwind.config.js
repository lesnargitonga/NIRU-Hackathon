/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'lesnar': {
          'primary': '#667eea',
          'secondary': '#764ba2',
          'accent': '#f093fb',
        }
      }
    },
  },
  plugins: [],
}
