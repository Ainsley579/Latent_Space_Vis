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

type JsmeInstance = {
  jmeFile?: () => string;
  readMolecule?: (input: string) => void;
};

const JSME_SCRIPT_ID = "jsme-editor-script";
const LOCAL_JSME_SCRIPT_SRC = "/jsme/jsme.nocache.js";

export default function StructureEditorPanel() {
  const reactId = useId();
  const hostIdRef = useRef(`jsme_modal_host_${reactId.replace(/:/g, "_")}`);
  const previewHostIdRef = useRef(`jsme_preview_host_${reactId.replace(/:/g, "_")}`);
  const previewEditorRef = useRef<JsmeInstance | null>(null);
  const modalEditorRef = useRef<JsmeInstance | null>(null);
  const previewInitedRef = useRef(false);
  const [isOpen, setIsOpen] = useState(false);
  const [scriptReady, setScriptReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let detachedLoadHandler: ((this: HTMLScriptElement, ev: Event) => void) | null = null;
    let detachedErrorHandler: ((this: HTMLScriptElement, ev: Event | string) => void) | null = null;
    const onReady = () => {
      if (cancelled) return;
      if (!window.JSApplet?.JSME) return;
      setScriptReady(true);
      setError(null);
    };
    window.jsmeOnLoad = onReady;

    const attachToExistingScript = (script: HTMLScriptElement) => {
      if (window.JSApplet?.JSME) {
        onReady();
        return;
      }
      detachedLoadHandler = () => {
        // Some JSME builds fire script "load" before JSApplet is fully initialized.
        // Only mark ready if constructor is already present; otherwise wait for jsmeOnLoad.
        if (window.JSApplet?.JSME) onReady();
      };
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
    script.onload = () => {
      // Keep this as a fallback only when JSApplet is already available at onload time.
      if (window.JSApplet?.JSME) onReady();
    };
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
    if (!scriptReady || previewInitedRef.current) return;
    const previewHost = document.getElementById(previewHostIdRef.current);
    if (!previewHost) return;
    previewHost.innerHTML = "";

    if (!window.JSApplet?.JSME) {
      setError("JSME failed to initialize.");
      return;
    }

    try {
      previewEditorRef.current = new window.JSApplet.JSME(
        previewHostIdRef.current,
        "100%",
        "100%",
      ) as JsmeInstance;
      previewInitedRef.current = true;
    } catch (e) {
      console.error(e);
      setError("Unable to load JSME preview.");
    }
  }, [scriptReady]);

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
      modalEditorRef.current = new window.JSApplet.JSME(hostIdRef.current, "100%", "100%") as JsmeInstance;
      const previewJme = previewEditorRef.current?.jmeFile?.();
      if (previewJme && modalEditorRef.current?.readMolecule) {
        modalEditorRef.current.readMolecule(previewJme);
      }
    } catch (e) {
      console.error(e);
      setError("Unable to load JSME editor.");
    }
  }, [isOpen, scriptReady]);

  function closeModalAndSync() {
    const modalJme = modalEditorRef.current?.jmeFile?.();
    if (modalJme && previewEditorRef.current?.readMolecule) {
      previewEditorRef.current.readMolecule(modalJme);
    }
    setIsOpen(false);
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div className="shrink-0 px-2 py-1 text-sm font-semibold text-zinc-700 border-b border-zinc-200">
        Structure Editor
      </div>
      <div className="flex-1 min-h-0 p-2">
        <button
          type="button"
          className="w-full h-full rounded border border-zinc-300 bg-white text-left overflow-hidden"
          onClick={() => setIsOpen(true)}
        >
          <div className="h-full min-h-0 flex flex-col">
            <div className="shrink-0 px-2 py-1 text-[11px] text-zinc-600 border-b border-zinc-200 flex items-center justify-between">
              <span>Preview</span>
              <span className="text-zinc-500">Click to edit</span>
            </div>
            <div className="flex-1 min-h-0 p-1 relative">
              <div
                id={previewHostIdRef.current}
                className="w-full h-full border border-zinc-200 rounded bg-white pointer-events-none"
              />
            </div>
          </div>
        </button>
        {error && (
          <div className="mt-1 text-xs text-amber-700">
            {error}
          </div>
        )}
      </div>
      {isOpen && (
        <div className="fixed inset-0 z-[9999] bg-black/40 flex items-center justify-center p-6">
          <div className="w-[min(1200px,92vw)] h-[82vh] bg-white border border-zinc-300 rounded shadow-xl flex flex-col">
            <div className="shrink-0 px-3 py-2 border-b border-zinc-200 flex items-center justify-between">
              <div className="text-sm font-semibold text-zinc-700">JSME Structure Editor</div>
              <button
                type="button"
                className="px-2 py-1 rounded border border-zinc-300 text-xs text-zinc-700 hover:bg-zinc-50"
                onClick={closeModalAndSync}
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
