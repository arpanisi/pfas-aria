import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

export function setAuthToken(getToken: () => Promise<string | null>) {
  apiClient.interceptors.request.use(async (config) => {
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
      window.location.href = "/sign-in";
    }
    return Promise.reject(error);
  }
);
