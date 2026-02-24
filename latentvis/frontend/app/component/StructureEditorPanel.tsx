"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

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
  smiles?: () => string;
  readMolecule?: (input: string) => void;
  svg?: () => string;
  toSVG?: () => string;
  svgFile?: () => string;
  setSize?: (width: string, height: string) => void;
};

const JSME_SCRIPT_ID = "jsme-editor-script";
const LOCAL_JSME_SCRIPT_SRC = "/jsme/jsme.nocache.js";

export default function StructureEditorPanel() {
  const reactId = useId();
  const inlineHostId = useMemo(() => `jsme_inline_host_${reactId.replace(/:/g, "_")}`, [reactId]);
  const hostId = useMemo(() => `jsme_modal_host_${reactId.replace(/:/g, "_")}`, [reactId]);
  const inlineEditorRef = useRef<JsmeInstance | null>(null);
  const modalEditorRef = useRef<JsmeInstance | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [scriptReady, setScriptReady] = useState(false);
  const [currentJme, setCurrentJme] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [searchBusy, setSearchBusy] = useState(false);

  function getHostSize(element: HTMLElement) {
    const width = Math.max(240, Math.floor(element.clientWidth));
    const height = Math.max(160, Math.floor(element.clientHeight));
    return { width: `${width}px`, height: `${height}px` };
  }

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
    if (!scriptReady) return;
    const host = document.getElementById(inlineHostId);
    if (!host) return;
    const JsmeCtor = window.JSApplet?.JSME;
    if (!JsmeCtor) return;

    const mountInlineEditor = () => {
      host.innerHTML = "";
      const { width, height } = getHostSize(host);
      try {
        inlineEditorRef.current = new JsmeCtor(inlineHostId, width, height) as JsmeInstance;
        if (currentJme && inlineEditorRef.current?.readMolecule) {
          inlineEditorRef.current.readMolecule(currentJme);
        }
      } catch (e) {
        console.error(e);
        window.setTimeout(() => setError("Unable to load JSME editor."), 0);
      }
    };

    mountInlineEditor();

    const observer = new ResizeObserver(() => {
      if (!inlineEditorRef.current?.setSize) return;
      const { width, height } = getHostSize(host);
      inlineEditorRef.current.setSize(width, height);
    });
    observer.observe(host);

    return () => {
      observer.disconnect();
    }
  }, [scriptReady, inlineHostId, currentJme]);

  useEffect(() => {
    if (!isOpen || !scriptReady) return;
    const host = document.getElementById(hostId);
    if (!host) return;
    host.innerHTML = "";

    if (!window.JSApplet?.JSME) {
      return;
    }

    try {
      const { width, height } = getHostSize(host);
      modalEditorRef.current = new window.JSApplet.JSME(hostId, width, height) as JsmeInstance;
      if (currentJme && modalEditorRef.current?.readMolecule) {
        modalEditorRef.current.readMolecule(currentJme);
      }
    } catch (e) {
      console.error(e);
      window.setTimeout(() => setError("Unable to load JSME editor."), 0);
    }
  }, [isOpen, scriptReady, currentJme, hostId]);

  useEffect(() => {
    if (!currentJme || !inlineEditorRef.current?.readMolecule) return;
    inlineEditorRef.current.readMolecule(currentJme);
  }, [currentJme]);

  function openModalFromInline() {
    const inlineJme = inlineEditorRef.current?.jmeFile?.();
    if (inlineJme) setCurrentJme(inlineJme);
    setIsOpen(true);
  }

  function closeModalAndSync() {
    const modalJme = modalEditorRef.current?.jmeFile?.();
    if (modalJme) setCurrentJme(modalJme);
    setIsOpen(false);
  }

  function getActiveSmiles() {
    const editor = isOpen ? modalEditorRef.current : inlineEditorRef.current;
    const editorSmiles = editor?.smiles?.();
    if (editorSmiles && editorSmiles.trim().length > 0) return editorSmiles.trim();
    return null;
  }

  async function searchCurrentSmiles() {
    const smiles = getActiveSmiles();
    if (!smiles) {
      setError("Current structure cannot be converted to SMILES.");
      return;
    }
    setSearchBusy(true);
    setError(null);
    try {
      const apiBase = process.env.NEXT_PUBLIC_FLASK_API_BASE || "http://127.0.0.1:5001";
      const res = await fetch(`${apiBase}/search-smiles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      });
      if (!res.ok) {
        throw new Error(`search-smiles API failed: ${res.status}`);
      }
    } catch (e) {
      console.error(e);
      setError("Failed to send SMILES to backend.");
    } finally {
      setSearchBusy(false);
    }
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div className="shrink-0 px-2 py-1 text-sm font-semibold text-zinc-700 border-b border-zinc-200">
        Structure Editor
      </div>
      <div className="flex-1 min-h-0 p-3 flex flex-col gap-2">
        <div className="relative h-48 w-full border border-zinc-200 rounded bg-white overflow-hidden">
          <button
            type="button"
            className={
              "absolute left-2 top-2 z-10 px-3 py-1 rounded border text-xs shadow-sm " +
              (searchBusy
                ? "border-zinc-200 bg-zinc-100 text-zinc-400 cursor-not-allowed"
                : "border-zinc-300 bg-white/95 text-zinc-700 hover:bg-white")
            }
            onClick={searchCurrentSmiles}
            disabled={searchBusy}
          >
            {searchBusy ? "Searching..." : "Search"}
          </button>
          <button
            type="button"
            className="absolute right-2 top-2 z-10 px-3 py-1 rounded border border-zinc-300 bg-white/95 text-xs text-zinc-700 shadow-sm hover:bg-white"
            onClick={openModalFromInline}
          >
            {currentJme ? "Open JSME (Edit)" : "Open JSME"}
          </button>
          <div id={inlineHostId} className="w-full h-full" />
        </div>
        {error && (
          <div className="text-xs text-amber-700">
            {error}
          </div>
        )}
      </div>
      {isOpen && (
        <div className="fixed inset-0 z-[9999] bg-black/40 flex items-center justify-center p-6">
          <div className="w-[min(1200px,92vw)] h-[82vh] bg-white border border-zinc-300 rounded shadow-xl flex flex-col">
            <div className="shrink-0 px-3 py-2 border-b border-zinc-200 flex items-center justify-between">
              <div className="text-sm font-semibold text-zinc-700">JSME Structure Editor</div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className={
                    "px-2 py-1 rounded border text-xs " +
                    (searchBusy
                      ? "border-zinc-200 bg-zinc-100 text-zinc-400 cursor-not-allowed"
                      : "border-zinc-300 text-zinc-700 hover:bg-zinc-50")
                  }
                  onClick={searchCurrentSmiles}
                  disabled={searchBusy}
                >
                  {searchBusy ? "Searching..." : "Search"}
                </button>
                <button
                  type="button"
                  className="px-2 py-1 rounded border border-zinc-300 text-xs text-zinc-700 hover:bg-zinc-50"
                  onClick={closeModalAndSync}
                >
                  Close
                </button>
              </div>
            </div>
            <div className="flex-1 min-h-0 p-2">
              <div id={hostId} className="w-full h-full border border-zinc-200 rounded bg-white" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
