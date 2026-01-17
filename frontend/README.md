# UMI Frontend

Next.js web application for the UMI Medical LLM Platform.

## Theme

- **Primary Color**: Violet (#7c3aed)
- **Background**: White (#ffffff)
- **Accent**: Purple gradients

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **HTTP Client**: Axios
- **Icons**: Lucide React
- **Forms**: React Hook Form + Zod

## Getting Started

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── page.tsx            # Landing page
│   │   ├── login/              # Login page
│   │   ├── register/           # Registration page
│   │   ├── dashboard/          # User dashboard
│   │   └── consultation/       # AI consultation chat
│   ├── components/             # Reusable components
│   │   └── providers.tsx       # React Query provider
│   └── lib/                    # Utilities
│       ├── api.ts              # API client
│       └── auth.ts             # Auth state management
├── tailwind.config.ts          # Tailwind configuration
└── package.json
```

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with features |
| `/login` | User login |
| `/register` | User registration |
| `/dashboard` | User dashboard with quick actions |
| `/consultation` | AI-powered health consultation |

## Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=UMI
```

## Building for Production

```bash
npm run build
npm start
```

## Connecting to Backend

The frontend connects to the UMI backend API. Make sure the backend is running:

```bash
# In the root directory
docker-compose up -d

# Or run directly
uvicorn src.main:app --reload
```

## Color Palette

```css
/* Violet Theme */
--primary-50: #f5f3ff;
--primary-100: #ede9fe;
--primary-200: #ddd6fe;
--primary-300: #c4b5fd;
--primary-400: #a78bfa;
--primary-500: #8b5cf6;
--primary-600: #7c3aed;  /* Primary */
--primary-700: #6d28d9;
--primary-800: #5b21b6;
--primary-900: #4c1d95;
```
