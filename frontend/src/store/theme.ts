import { create } from "zustand";

const STORAGE_KEY = "pfas-aria-theme";

const savedDark = localStorage.getItem(STORAGE_KEY);
const initialDark = savedDark !== null ? savedDark === "dark" : true;
document.documentElement.setAttribute("data-theme", initialDark ? "dark" : "light");

interface ThemeStore {
  dark: boolean;
  toggle: () => void;
}

export const useTheme = create<ThemeStore>(() => ({
  dark: initialDark,
  toggle: () =>
    useTheme.setState((s) => {
      const next = !s.dark;
      document.documentElement.setAttribute("data-theme", next ? "dark" : "light");
      localStorage.setItem(STORAGE_KEY, next ? "dark" : "light");
      return { dark: next };
    }),
}));
