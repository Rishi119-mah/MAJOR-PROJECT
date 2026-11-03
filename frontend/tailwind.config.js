/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef2ff',
          100: '#e0e7ff',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
        },
        sidebar: {
          bg: '#0f172a',
          surface: '#1e293b',
          border: '#334155',
          text: '#f8fafc',
          muted: '#94a3b8',
          hover: '#1e293b',
          active: '#334155',
        },
        risk: {
          low: '#10b981',
          lowBg: '#ecfdf5',
          lowBorder: '#a7f3d0',
          mod: '#f59e0b',
          modBg: '#fffbeb',
          modBorder: '#fde68a',
          high: '#ef4444',
          highBg: '#fef2f2',
          highBorder: '#fecaca',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      }
    },
  },
  plugins: [],
}

