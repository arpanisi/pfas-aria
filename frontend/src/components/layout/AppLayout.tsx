import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { useTheme } from "@/store/theme";
import { useEffect } from "react";

export function AppLayout() {
  const { dark } = useTheme();

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }, [dark]);

  return (
    <div className="app-shell">
      <Sidebar />
      <main style={{ flex: 1, overflow: "hidden", display: "contents" }}>
        <Outlet />
      </main>
    </div>
  );
}
