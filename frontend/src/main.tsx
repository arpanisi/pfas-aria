import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import {
  ClerkProvider,
  SignIn,
  SignedIn,
  SignedOut,
  useAuth,
} from "@clerk/clerk-react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";

import { AppLayout } from "@/components/layout/AppLayout";
import { Dashboard } from "@/pages/Dashboard";
import { NewRun } from "@/pages/NewRun";
import { RunHistory } from "@/pages/RunHistory";
import { CorpusPage } from "@/pages/CorpusPage";
import { useTheme } from "@/store/theme";
import { setAuthToken } from "@/api/client";

import "./index.css";

const CLERK_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY as string;

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

function ThemeInit() {
  const { dark } = useTheme();
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }, [dark]);
  return null;
}

function AuthSetup() {
  const { getToken } = useAuth();
  useEffect(() => {
    setAuthToken(() => getToken());
  }, [getToken]);
  return null;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={CLERK_KEY}>
      <QueryClientProvider client={queryClient}>
        <ThemeInit />
        <BrowserRouter>
          <SignedOut>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100vh",
                background: "var(--bg)",
              }}
            >
              <SignIn />
            </div>
          </SignedOut>
          <SignedIn>
            <AuthSetup />
            <Routes>
              <Route element={<AppLayout />}>
                <Route index element={<Navigate to="/runs" replace />} />
                <Route path="/sign-in" element={<Navigate to="/runs" replace />} />
                <Route path="/runs" element={<RunHistory />} />
                <Route path="/runs/:runId" element={<Dashboard />} />
                <Route path="/upload" element={<NewRun />} />
                <Route path="/corpus" element={<CorpusPage />} />
                <Route path="*" element={<Navigate to="/runs" replace />} />
              </Route>
            </Routes>
          </SignedIn>
        </BrowserRouter>
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: "var(--surface)",
              color: "var(--text)",
              border: "1px solid var(--border)",
              fontFamily: "var(--font-sans)",
              fontSize: "13px",
            },
          }}
        />
      </QueryClientProvider>
    </ClerkProvider>
  </React.StrictMode>
);
