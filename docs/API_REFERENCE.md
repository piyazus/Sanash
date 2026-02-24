# Sanash API Reference

**Base URL:** `http://localhost:8000/api/v1`
**Authentication:** JWT Bearer token (except public endpoints)
**Content-Type:** `application/json`

---

## Authentication

### POST `/auth/register`
Register a new user account.

**Request:**
```json
{ "username": "operator1", "email": "op@example.com", "password": "secure123" }
```
**Response (201):**
```json
{ "id": 1, "username": "operator1", "email": "op@example.com", "role": "operator" }
```

---

### POST `/auth/login`
Obtain a JWT access token.

**Request:**
```json
{ "username": "operator1", "password": "secure123" }
```
**Response (200):**
```json
{ "access_token": "eyJ...", "token_type": "bearer", "expires_in": 1800 }
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"operator1","password":"secure123"}'
```

---

### POST `/auth/refresh`
Refresh an expiring access token.

**Auth:** Required
**Response (200):** New `{ "access_token": "..." }`

---

### GET `/auth/me`
Get current authenticated user profile.

**Auth:** Required
**Response (200):**
```json
{ "id": 1, "username": "operator1", "role": "operator", "created_at": "2025-01-01T00:00:00" }
```

---

## Buses

### GET `/buses`
List all tracked buses with current occupancy.

**Auth:** Required
**Response (200):**
```json
[
  {
    "bus_id": "bus_001",
    "route_id": "36A",
    "current_count": 35,
    "capacity": 60,
    "occupancy_ratio": 0.583,
    "status": "Yellow",
    "last_updated": "2025-03-15T08:30:00Z"
  }
]
```

---

### GET `/buses/{bus_id}`
Get detailed information for a specific bus.

**Auth:** Required
**Path:** `bus_id` — Bus identifier (e.g., `bus_001`)
**Response (200):** Single bus object (same schema as above)

---

### GET `/buses/{bus_id}/occupancy`
Get current occupancy for a bus.

**Auth:** Required
**Response (200):**
```json
{
  "bus_id": "bus_001",
  "count": 35,
  "capacity": 60,
  "occupancy_ratio": 0.583,
  "status": "Yellow",
  "status_color": "#FFC107",
  "timestamp": "2025-03-15T08:30:00Z"
}
```

Status values: `"Green"` (≤50%), `"Yellow"` (51–80%), `"Red"` (>80%)

---

### POST `/buses/{bus_id}/count`
Submit a new crowd count reading from an edge device (Jetson Nano).

**Auth:** Required
**Request:**
```json
{ "count": 42, "camera_id": "cam_front", "confidence": 0.94, "timestamp": "2025-03-15T08:30:00Z" }
```
**Response (201):**
```json
{ "id": 1234, "bus_id": "bus_001", "count": 42, "status": "Yellow", "recorded_at": "2025-03-15T08:30:01Z" }
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/buses/bus_001/count \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{"count":42,"camera_id":"cam_front","confidence":0.94}'
```

---

## Cameras

### GET `/cameras`
List all registered cameras.

**Auth:** Required
**Response (200):** `[{ "camera_id": "cam_001", "bus_id": "bus_001", "status": "active" }]`

---

### POST `/cameras/{camera_id}/snapshot`
Trigger on-demand inference on the current camera frame.

**Auth:** Required
**Response (202):** `{ "job_id": "job_abc123", "status": "queued" }`

---

## Analytics

### GET `/analytics/daily`
Daily aggregated occupancy statistics.

**Auth:** Required
**Query:** `?date=2025-03-15&route_id=36A`
**Response (200):**
```json
{
  "date": "2025-03-15",
  "avg_occupancy": 0.64,
  "peak_occupancy": 0.97,
  "total_readings": 2880,
  "busiest_hour": 8
}
```

---

### GET `/analytics/hourly`
Hourly occupancy breakdown for a bus or route.

**Auth:** Required
**Query:** `?bus_id=bus_001&date=2025-03-15`
**Response (200):** `[{ "hour": 8, "avg_occupancy": 0.87, "readings": 120 }, ...]`

---

### GET `/analytics/heatmap`
Occupancy heatmap data for dashboard visualization.

**Auth:** Required
**Query:** `?date=2025-03-15`
**Response (200):** Matrix of occupancy by hour × route.

---

## Zones (Bus Stops)

### GET `/zones`
List all bus stop zones.

**Auth:** Required
**Response (200):** `[{ "zone_id": "stop_01", "name": "Alatau", "lat": 43.25, "lon": 76.91 }]`

---

### GET `/zones/{zone_id}/stats`
Entry/exit counts for a bus stop zone.

**Auth:** Required
**Response (200):**
```json
{ "zone_id": "stop_01", "entries_today": 1240, "exits_today": 1190, "avg_wait_min": 3.2 }
```

---

## Jobs

### GET `/jobs`
List active and recent background processing jobs.

**Auth:** Required
**Response (200):** `[{ "job_id": "abc123", "type": "detection", "status": "running", "progress": 0.6 }]`

---

### GET `/jobs/{job_id}`
Get status of a specific job.

**Auth:** Required
**Response (200):** `{ "job_id": "abc123", "status": "completed", "result": {...} }`

---

## Alerts

### GET `/alerts`
List recent overcrowding alerts.

**Auth:** Required
**Query:** `?limit=20&unread_only=true`
**Response (200):**
```json
[{ "alert_id": 1, "bus_id": "bus_001", "type": "overcrowding", "occupancy": 0.95,
   "created_at": "2025-03-15T08:32:00Z", "acknowledged": false }]
```

---

### POST `/alerts/acknowledge/{alert_id}`
Acknowledge an alert.

**Auth:** Required
**Response (200):** `{ "alert_id": 1, "acknowledged": true }`

---

## Public Endpoints (No Auth)

### GET `/public/buses`
Public bus list showing only status (no raw counts for privacy).

**Auth:** None required
**Response (200):**
```json
[{ "bus_id": "bus_001", "route_id": "36A", "status": "Yellow", "last_updated": "..." }]
```

---

### GET `/public/stops`
List of bus stops with coordinates.

**Auth:** None required
**Response (200):** `[{ "stop_id": "stop_01", "name": "Alatau", "lat": 43.25, "lon": 76.91 }]`

---

## Mobile

### GET `/mobile/nearby`
Get buses and stops near a GPS coordinate.

**Auth:** Required
**Query:** `?lat=43.25&lon=76.91&radius=500`
**Response (200):**
```json
{
  "buses": [{ "bus_id": "bus_001", "status": "Green", "distance_m": 120 }],
  "stops": [{ "stop_id": "stop_01", "name": "Alatau", "distance_m": 50 }]
}
```

---

## Health

### GET `/health`
System health check (no auth required).

**Response (200):**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "connected",
  "cache": "connected",
  "uptime_seconds": 86400
}
```

Status values: `"ok"`, `"degraded"`, `"unhealthy"`
