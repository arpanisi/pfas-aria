import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

// Inject Clerk token on every request
apiClient.interceptors.request.use(async (config) => {
  try {
    delete config.headers.Authorization;
  } catch {
    // No token — public route
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      window.location.href = "/sign-in";
    }
    return Promise.reject(error);
  }
);
