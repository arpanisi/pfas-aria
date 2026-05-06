import { NavLink } from "react-router-dom";
import {
  Activity,
  BookOpen,
  Database,
  FlaskConical,
  LayoutDashboard,
} from "lucide-react";
import clsx from "clsx";

const NAV = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/runs", icon: Activity, label: "Runs" },
  { to: "/upload", icon: FlaskConical, label: "New Run" },
  { to: "/corpus", icon: BookOpen, label: "Corpus" },
  { to: "/data", icon: Database, label: "Data" },
];

export function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-text">PFAS</span>
        <span className="logo-accent">ARIA</span>
      </div>

      <nav className="sidebar-nav">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              clsx("nav-item", { "nav-item--active": isActive })
            }
          >
            <Icon size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="version-badge">v0.1.0</div>
      </div>
    </aside>
  );
}
