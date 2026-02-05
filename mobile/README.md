# Sana Bus - React Native Mobile App

Real-time bus occupancy monitoring for Almaty passengers.

## Features

- 🗺️ Full-screen interactive map
- 🚌 Real-time bus positions with occupancy
- 🚏 Bus stop arrivals with ETAs
- 🔍 Search by route or bus number
- 📴 Offline mode support

## Setup

1. Install dependencies:
```bash
cd mobile
npm install
```

2. Add Google Maps API key in `app.json`:
   - `android.config.googleMaps.apiKey`
   - `ios.config.googleMapsApiKey`

3. Start the app:
```bash
npm start
```

4. Run on device:
```bash
npm run android  # or npm run ios
```

## Project Structure

```
mobile/
├── App.tsx              # Entry point
├── src/
│   ├── components/      # Reusable components
│   │   ├── BusMarker.tsx
│   │   ├── StopMarker.tsx
│   │   ├── StopDetailSheet.tsx
│   │   ├── SearchBar.tsx
│   │   └── OfflineIndicator.tsx
│   ├── screens/
│   │   └── MapScreen.tsx
│   ├── hooks/
│   │   └── useData.ts
│   ├── services/
│   │   └── api.ts
│   └── store/
│       └── useAppStore.ts
```

## API Endpoints Used

- `GET /api/v1/mobile/buses/positions` - All bus GPS positions
- `GET /api/v1/mobile/stops` - All bus stops
- `GET /api/v1/mobile/stops/{id}/arrivals` - ETAs at stop
