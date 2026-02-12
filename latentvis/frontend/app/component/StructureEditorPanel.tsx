"use client";

import { useEffect, useId, useRef, useState } from "react";

declare global {
  interface Window {
    JSApplet?: {
      JSME: new (elementId: string, width: string, height: string, options?: string) => unknown;
    };
    jsmeOnLoad?: () => void;
  }
}

const JSME_SCRIPT_ID = "jsme-editor-script";
const LOCAL_JSME_SCRIPT_SRC = "/jsme/jsme.nocache.js";

export default function StructureEditorPanel() {
  const reactId = useId();
  const hostIdRef = useRef(`jsme_modal_host_${reactId.replace(/:/g, "_")}`);
  const [isOpen, setIsOpen] = useState(false);
  const [scriptReady, setScriptReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let detachedLoadHandler: ((this: HTMLScriptElement, ev: Event) => void) | null = null;
    let detachedErrorHandler: ((this: HTMLScriptElement, ev: Event | string) => void) | null = null;
    const onReady = () => {
      if (cancelled) return;
      setScriptReady(true);
      setError(null);
    };
    window.jsmeOnLoad = onReady;

    const attachToExistingScript = (script: HTMLScriptElement) => {
      if (window.JSApplet?.JSME) {
        onReady();
        return;
      }
      detachedLoadHandler = () => onReady();
      detachedErrorHandler = () => {
        if (cancelled) return;
        setError("Failed to load JSME script. Put JSME dist files under public/jsme.");
      };
      script.addEventListener("load", detachedLoadHandler, { once: true });
      script.addEventListener("error", detachedErrorHandler as EventListener, { once: true });
    };

    const existing = document.getElementById(JSME_SCRIPT_ID) as HTMLScriptElement | null;
    if (existing) {
      attachToExistingScript(existing);
      return () => {
        cancelled = true;
        if (window.jsmeOnLoad === onReady) {
          window.jsmeOnLoad = undefined;
        }
        if (detachedLoadHandler) existing.removeEventListener("load", detachedLoadHandler);
        if (detachedErrorHandler) existing.removeEventListener("error", detachedErrorHandler as EventListener);
      };
    }

    const script = document.createElement("script");
    script.id = JSME_SCRIPT_ID;
    script.async = true;
    script.src = LOCAL_JSME_SCRIPT_SRC;
    script.onload = () => onReady();
    script.onerror = () => {
      if (cancelled) return;
      setError("Failed to load local JSME script from /jsme/jsme.nocache.js.");
    };
    document.body.appendChild(script);

    return () => {
      cancelled = true;
      if (window.jsmeOnLoad === onReady) {
        window.jsmeOnLoad = undefined;
      }
      script.onload = null;
      script.onerror = null;
    };
  }, []);

  useEffect(() => {
    if (!isOpen || !scriptReady) return;
    const host = document.getElementById(hostIdRef.current);
    if (!host) return;
    host.innerHTML = "";

    if (!window.JSApplet?.JSME) {
      setError("JSME failed to initialize.");
      return;
    }

    try {
      setError(null);
      new window.JSApplet.JSME(hostIdRef.current, "100%", "100%");
    } catch (e) {
      console.error(e);
      setError("Unable to load JSME editor.");
    }
  }, [isOpen, scriptReady]);

  return (
    <div className="w-full h-full flex flex-col">
      <div className="shrink-0 px-2 py-1 text-sm font-semibold text-zinc-700 border-b border-zinc-200">
        Structure Editor
      </div>
      <div className="flex-1 min-h-0 p-2 flex items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            className="px-4 py-2 rounded border border-zinc-300 bg-white text-sm font-medium text-zinc-700 hover:bg-zinc-50"
            onClick={() => setIsOpen(true)}
          >
            Open Structure Editor
          </button>
          {error && <div className="text-xs text-amber-700">{error}</div>}
        </div>
      </div>
      {isOpen && (
        <div className="fixed inset-0 z-[9999] bg-black/40 flex items-center justify-center p-6">
          <div className="w-[min(1200px,92vw)] h-[82vh] bg-white border border-zinc-300 rounded shadow-xl flex flex-col">
            <div className="shrink-0 px-3 py-2 border-b border-zinc-200 flex items-center justify-between">
              <div className="text-sm font-semibold text-zinc-700">JSME Structure Editor</div>
              <button
                type="button"
                className="px-2 py-1 rounded border border-zinc-300 text-xs text-zinc-700 hover:bg-zinc-50"
                onClick={() => setIsOpen(false)}
              >
                Close
              </button>
            </div>
            <div className="flex-1 min-h-0 p-2">
              <div id={hostIdRef.current} className="w-full h-full border border-zinc-200 rounded bg-white" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
