import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

export const apiClient = axios.create({
  baseURL: BASE_URL,
});

export function apiErrorMessage(error: unknown, fallback = "Request failed") {
  if (!axios.isAxiosError(error)) return fallback;
  const detail = error.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail) {
    return String(detail.message);
  }
  return error.message || fallback;
}

let authInterceptor: number | null = null;

export function setAuthToken(getToken: () => Promise<string | null>) {
  // Remove previous interceptor to avoid stacking
  if (authInterceptor !== null) {
    apiClient.interceptors.request.eject(authInterceptor);
  }
  authInterceptor = apiClient.interceptors.request.use(async (config) => {
    const token = await getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  });
}

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Don't redirect — let Clerk handle auth state
      console.warn("401 Unauthorized — token may be expired");
    }
    return Promise.reject(error);
  }
);
