import React, { useEffect, useState } from "react";
import { createRoot, Root } from "react-dom/client";
import SankeyDiagram from "../../frontend/components/SankeyDiagram";

type AppProps = {
  initialData?: any[];
  dataUrl?: string;
};

export default function App({ initialData, dataUrl }: AppProps) {
  const [data, setData] = useState<any[]>(initialData ?? []);

  useEffect(() => {
    if ((initialData && initialData.length > 0) || (window as any).__SANKEY_DATA__) {
      if (!(initialData && initialData.length > 0) && (window as any).__SANKEY_DATA__) {
        setData((window as any).__SANKEY_DATA__);
      }
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const resolvedDataUrl = dataUrl || params.get("data") || "/sankey_notebook.json";
    fetch(resolvedDataUrl)
      .then((r) => r.json())
      .then((rows) => setData(rows))
      .catch((err) => {
        console.error("load data", err);
        setData([]);
      });
  }, [initialData, dataUrl]);

  return (
    <div style={{ padding: 8 }}>
      <h3>Sankey (Notebook)</h3>
      <SankeyDiagram data={data} zoomFactor={1} />
    </div>
  );
}

// Keep a global registry of mounted roots so we can support multiple mounts and cleanly unmount
declare global {
  interface Window {
    mountApp?: (elemId?: string, opts?: { initialData?: any[]; dataUrl?: string }) => { unmount: () => void };
    NotebookSankey?: any;
  }
}

function ensureContainer(elemId: string) {
  let container = document.getElementById(elemId);
  if (!container) {
    container = document.createElement("div");
    container.id = elemId;
    // append to the Jupyter output area if available, else body
    const outputArea = document.querySelector('.jp-OutputArea-output') as HTMLElement | null;
    (outputArea || document.body || document.documentElement).appendChild(container);
  }
  // clear previous contents to avoid duplication
  container.innerHTML = "";
  return container;
}

export function mountApp(elemId = "root", opts?: { initialData?: any[]; dataUrl?: string }) {
  // create registry if not present
  if (!window.NotebookSankey) window.NotebookSankey = {};
  if (!window.NotebookSankey._roots) window.NotebookSankey._roots = {};

  // if a root already exists for this elemId, unmount it first
  const existing: Root | undefined = window.NotebookSankey._roots[elemId];
  if (existing) {
    try {
      existing.unmount();
    } catch (e) {
      // continue regardless
      console.warn('previous root unmount failed', e);
    }
    delete window.NotebookSankey._roots[elemId];
  }

  const container = ensureContainer(elemId);
  const root = createRoot(container);

  const initialData = opts?.initialData ?? (window as any).__SANKEY_DATA__ ?? [];
  const dataUrl = opts?.dataUrl;

  root.render(<App initialData={initialData} dataUrl={dataUrl} />);

  // store root so we can unmount later
  window.NotebookSankey._roots[elemId] = root;

  return {
    unmount: () => {
      try {
        root.unmount();
        delete window.NotebookSankey._roots[elemId];
        const el = document.getElementById(elemId);
        if (el && el.parentNode) el.parentNode.removeChild(el);
      } catch (e) {
        console.warn('unmount failed', e);
      }
    },
  };
}

// convenience to mount many sankeys at once
export function mountMany(specs: { elemId: string; data?: any[]; dataUrl?: string }[]) {
  return specs.map((s) => mountApp(s.elemId, { initialData: s.data, dataUrl: s.dataUrl }));
}

// expose on window for IIFE bundles
window.mountApp = mountApp;
window.NotebookSankey = window.NotebookSankey || {};
window.NotebookSankey.mountApp = mountApp;
window.NotebookSankey.mountMany = mountMany;