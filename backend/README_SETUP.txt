ENHANCED MARKOV CHAIN–BASED VIEWER LOYALTY MANAGEMENT SYSTEM Frontend +
Backend Setup Guide

SYSTEM REQUIREMENTS - Python 3.10+ - Node.js 18+ - npm 9+ - Windows
PowerShell or CMD

PROJECT STRUCTURE project-root/ 
│ ├── backend/ │ 
	├── api_server.py │ 
	└── other backend files │ 
└── frontend/ 
	├── src/ 	
	├── package.json 
	└── vite.config.ts

FIRST TIME SETUP

1)  BACKEND SETUP

Open Terminal 1: cd project-root/backend

Install dependencies: pip install fastapi uvicorn python-multipart
pandas

Run backend server: python -m uvicorn api_server:app –reload –port 8000

Backend URL: http://localhost:8000

2)  FRONTEND SETUP

Open Terminal 2: cd project-root npm create vite@latest frontend –
–template react-ts cd frontend npm install

INSTALL FRONTEND DEPENDENCIES

npm install react-router-dom npm install lucide-react npm install
recharts npm install papaparse npm install chart.js react-chartjs-2 npm
install clsx tailwind-merge class-variance-authority npm install
@radix-ui/react-slot npm install @radix-ui/react-dropdown-menu

INSTALL TAILWIND CSS v3 (STABLE)

npm install -D tailwindcss@3.4.4 postcss autoprefixer npx tailwindcss
init -p

If npx fails: node ./node_modules/tailwindcss/lib/cli.js init -p

CONFIGURE tailwind.config.js

Replace contents with:

export default { content: [ “./index.html”,
“./src/**/*.{js,ts,jsx,tsx}“, ], theme: { extend: {}, }, plugins: [], }

CONFIGURE postcss.config.js

If using ESM:

export default { plugins: { tailwindcss: {}, autoprefixer: {}, }, }

If error occurs, rename file to postcss.config.cjs and use:

module.exports = { plugins: { tailwindcss: {}, autoprefixer: {}, }, }

CONFIGURE CSS

Open frontend/src/index.css Delete everything and add:

@tailwind base; @tailwind components; @tailwind utilities;

ENSURE CSS IMPORT

Open frontend/src/main.tsx and confirm:

import “./index.css”;

RUN FRONTEND

cd frontend npm run dev

Frontend URL: http://localhost:5173

RUNNING BOTH SERVERS

Terminal 1 -> Backend (Port 8000) Terminal 2 -> Frontend (Port 5173)

DEVELOPMENT WORKFLOW

Start backend: cd backend python -m uvicorn api_server:app –reload –port
8000

Start frontend: cd frontend npm run dev

TROUBLESHOOTING

If Tailwind styles are not applied: 
1. Ensure tailwindcss version is 3.4.4 
2. Ensure index.css contains only Tailwind directives 
3. Ensure main.tsx imports index.css 
4. Delete Vite cache: rmdir /s /q node_modules.vite 
5. Restart dev server

PRODUCTION BUILD

Frontend: cd frontend npm run build

Backend: python -m uvicorn api_server:app –reload –port 8000
