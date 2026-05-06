import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

// Inject Clerk token on every request
apiClient.interceptors.request.use(async (config) => {
  try {
    const { getToken } = await import("@clerk/clerk-react").then(
      (m) => m.useAuth()
    );
    // Note: useAuth() is called in components — token injected via store
    const token = sessionStorage.getItem("aria_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
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
