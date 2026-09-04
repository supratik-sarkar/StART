import React, { useEffect, useRef, useState } from "react";
import { ShieldCheck, RefreshCw, AlertCircle } from "lucide-react";

interface TurnstileWidgetProps {
  onTokenReceived: (token: string) => void;
  onError?: (err: string) => void;
  className?: string;
}

declare global {
  interface Window {
    turnstile?: {
      render: (
        container: string | HTMLElement,
        options: {
          sitekey: string;
          callback: (token: string) => void;
          "error-callback"?: (err: any) => void;
          "expired-callback"?: () => void;
          theme?: "light" | "dark" | "auto";
          size?: "normal" | "compact" | "flexible";
          mode?: "managed" | "non-interactive" | "invisible";
        }
      ) => string;
      reset: (widgetId: string) => void;
      remove: (widgetId: string) => void;
    };
  }
}

export const TurnstileWidget: React.FC<TurnstileWidgetProps> = ({
  onTokenReceived,
  onError,
  className = "",
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const widgetIdRef = useRef<string | null>(null);
  const [status, setStatus] = useState<"INITIALIZING" | "VERIFIED" | "ERROR">("INITIALIZING");
  const [token, setToken] = useState<string | null>(null);

  // Read sitekey strictly from environment configuration
  const siteKey = (import.meta as any).env?.VITE_TURNSTILE_SITE_KEY || "";

  useEffect(() => {
    let checkInterval: any = null;
    let isMounted = true;

    if (!siteKey) {
      setStatus("ERROR");
      if (onError) onError("VITE_TURNSTILE_SITE_KEY unconfigured");
      return;
    }

    const renderWidget = () => {
      if (!window.turnstile || !containerRef.current) return;
      if (widgetIdRef.current) {
        try {
          window.turnstile.remove(widgetIdRef.current);
        } catch {}
      }

      try {
        const id = window.turnstile.render(containerRef.current, {
          sitekey: siteKey,
          theme: "light",
          size: "normal",
          mode: "non-interactive",
          callback: (receivedToken: string) => {
            if (!isMounted) return;
            setToken(receivedToken);
            setStatus("VERIFIED");
            onTokenReceived(receivedToken);
          },
          "error-callback": (err: any) => {
            if (!isMounted) return;
            console.warn("Turnstile challenge error", err);
            setStatus("ERROR");
            if (onError) onError("Turnstile verification error");
          },
          "expired-callback": () => {
            if (!isMounted) return;
            setToken(null);
            setStatus("INITIALIZING");
            if (widgetIdRef.current && window.turnstile) {
              window.turnstile.reset(widgetIdRef.current);
            }
          },
        });
        widgetIdRef.current = id;
      } catch (e: any) {
        console.warn("Failed to render Turnstile widget", e);
        if (isMounted) setStatus("ERROR");
      }
    };

    if (window.turnstile) {
      renderWidget();
    } else {
      checkInterval = setInterval(() => {
        if (window.turnstile) {
          clearInterval(checkInterval);
          renderWidget();
        }
      }, 200);
    }

    return () => {
      isMounted = false;
      if (checkInterval) clearInterval(checkInterval);
      if (widgetIdRef.current && window.turnstile) {
        try {
          window.turnstile.remove(widgetIdRef.current);
        } catch {}
      }
    };
  }, [siteKey]);

  return (
    <div className={`flex flex-col items-start gap-1.5 ${className}`}>
      <div className="flex items-center gap-2 text-xs font-medium text-stone-600">
        <ShieldCheck className="w-3.5 h-3.5 text-indigo-600" />
        <span>Execution Gate Verification</span>
        {status === "VERIFIED" && (
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-emerald-50 text-emerald-700 border border-emerald-200">
            VERIFIED
          </span>
        )}
        {status === "INITIALIZING" && (
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-stone-100 text-stone-600 border border-stone-200">
            CONNECTING...
          </span>
        )}
      </div>

      <div ref={containerRef} className="min-h-[65px] flex items-center" />

      {status === "ERROR" && (
        <div className="flex items-center gap-1.5 text-xs text-rose-600 bg-rose-50 px-2 py-1 rounded border border-rose-200">
          <AlertCircle className="w-3.5 h-3.5" />
          <span>Verification challenge failed. Please refresh or retry.</span>
        </div>
      )}
    </div>
  );
};
