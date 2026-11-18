# REST API Documentation - AI Fake News Detection System

Base URL: `http://127.0.0.1:5000/api`

---

## 1. Health Check
Retrieves the operational status of the Flask backend, database connectivity, and the inference engine.

- **Endpoint:** `GET /api/health`
- **Authentication:** None (Public)
- **Response (200 OK):**
```json
{
  "status": "ok",
  "database": "offline",
  "inference_engine": "ready",
  "supported_models": [
    "logistic_regression",
    "random_forest",
    "svm"
  ],
  "timestamp": "2026-11-19T00:30:00Z"
}
```

---

## 2. Analyze News (Prediction)
Analyzes an article or headline using the specified machine learning model and returns the predicted label, confidence percentage, and class probabilities.

- **Endpoint:** `POST /api/news/analyze` (Legacy alias: `POST /api/predict`)
- **Authentication:** Optional. If authenticated via session cookie, the prediction is logged to the user's prediction history. If MySQL is offline, the prediction is still executed and returned with `db_saved: false`.
- **Rate Limit:** 60 requests per minute per IP.
- **Request Headers:**
  `Content-Type: application/json`
- **Request Body:**
```json
{
  "text": "The government released its official economic report detailing fiscal improvements.",
  "model": "logistic_regression"
}
```
  - `text` *(required, string)*: 10 to 50,000 characters.
  - `model` *(optional, string)*: One of `logistic_regression`, `random_forest`, `svm`. Default is `logistic_regression`.

- **Success Response (200 OK):**
```json
{
  "success": true,
  "prediction": 1,
  "prediction_label": "REAL",
  "confidence": 88.5,
  "fake_probability": 11.5,
  "real_probability": 88.5,
  "model_key": "logistic_regression",
  "model_used": "Logistic Regression",
  "is_fallback": false,
  "db_saved": false,
  "timestamp": "2026-11-19T00:30:00.123456Z"
}
```

- **Error Responses:**
  - `400 Bad Request` (Missing text or invalid model):
    ```json
    {
      "success": false,
      "error": "News text is required.",
      "code": "MISSING_TEXT"
    }
    ```
  - `429 Too Many Requests` (Rate limit exceeded):
    ```json
    {
      "success": false,
      "error": "Rate limit exceeded. Please wait before submitting more requests.",
      "code": "RATE_LIMIT_EXCEEDED",
      "retry_after": 45
    }
    ```

---

## 3. Prediction History
Retrieves paginated prediction audit history for the authenticated user.

- **Endpoint:** `GET /api/news/history` (Legacy alias: `GET /api/history`)
- **Authentication:** Required (User session cookie).
- **Query Parameters:**
  - `page` *(optional, integer, default: 1)*
  - `per_page` *(optional, integer, default: 10, max: 100)*
- **Success Response (200 OK):**
```json
{
  "success": true,
  "page": 1,
  "per_page": 10,
  "total": 24,
  "total_pages": 3,
  "rows": [
    {
      "id": 1,
      "model_name": "Logistic Regression",
      "prediction_label": "REAL",
      "confidence": 92.4,
      "input_type": "API",
      "created_at": "2026-11-19 00:25:00",
      "preview": "The treasury department issued a statement regarding national economic forecasts today."
    }
  ]
}
```
- **Error Responses:**
  - `401 Unauthorized` (Unauthenticated request):
    ```json
    {
      "success": false,
      "error": "Authentication required. Please log in first.",
      "code": "UNAUTHORIZED"
    }
    ```
  - `503 Service Unavailable` (MySQL database offline):
    ```json
    {
      "success": false,
      "error": "Database service is currently unreachable.",
      "code": "DATABASE_OFFLINE",
      "rows": [],
      "total": 0
    }
    ```

---

## 4. User Registration
Registers a new user account with bcrypt password hashing.

- **Endpoint:** `POST /api/auth/register` (Legacy alias: `POST /api/register`)
- **Authentication:** None (Public).
- **Rate Limit:** 10 requests per minute per IP.
- **Request Body:**
```json
{
  "username": "student_dev",
  "email": "student@example.com",
  "password": "Password123!",
  "confirm_password": "Password123!"
}
```
- **Success Response (201 Created):**
```json
{
  "success": true,
  "message": "User registered successfully. You may now log in."
}
```

---

## 5. User Authentication / Login
Authenticates an existing user and establishes an HTTP session cookie.

- **Endpoint:** `POST /api/auth/login` (Legacy alias: `POST /api/login`)
- **Authentication:** None (Public).
- **Rate Limit:** 15 requests per minute per IP.
- **Request Body:**
```json
{
  "username": "student_dev",
  "password": "Password123!"
}
```
- **Success Response (200 OK):**
```json
{
  "success": true,
  "message": "Login successful.",
  "user": {
    "id": 1,
    "username": "student_dev",
    "email": "student@example.com",
    "role": "user"
  }
}
```
