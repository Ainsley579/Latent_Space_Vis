"use client";

import { useEffect, useState } from "react";

type SmilesStructureProps = {
  smiles: string | null | undefined;
  width?: number;
  height?: number;
  className?: string;
};

const OCL_CDN =
  "https://www.lactame.com/lib/openchemlib/5.2.0/openchemlib-minimal.js";
const OCL_LOAD_TIMEOUT_MS = 3500;
const SMILES_DRAWER_TIMEOUT_MS = 2500;
const SMILES_SVG_CACHE_LIMIT = 1200;
const OCL_RETRY_COOLDOWN_MS = 30000;

let oclLoadPromise: Promise<any> | null = null;
let smilesDrawerLoadPromise: Promise<any> | null = null;
const smilesSvgCache = new Map<string, string>();
const smilesSvgInflight = new Map<string, Promise<string | null>>();
let lastOclLoadFailureAt = 0;

function rememberSvgCache(key: string, svg: string) {
  if (!key || !svg) return;
  if (smilesSvgCache.has(key)) smilesSvgCache.delete(key);
  smilesSvgCache.set(key, svg);
  if (smilesSvgCache.size <= SMILES_SVG_CACHE_LIMIT) return;
  const oldest = smilesSvgCache.keys().next().value;
  if (oldest) smilesSvgCache.delete(oldest);
}

function loadOCL(): Promise<any> {
  if (typeof window === "undefined") return Promise.reject(new Error("no_window"));
  const w = window as any;
  if (w.OCL) return Promise.resolve(w.OCL);
  if (Date.now() - lastOclLoadFailureAt < OCL_RETRY_COOLDOWN_MS) {
    return Promise.reject(new Error("ocl_retry_cooldown"));
  }
  if (oclLoadPromise) return oclLoadPromise;

  oclLoadPromise = new Promise((resolve, reject) => {
    const settleWithWindowOcl = () => {
      const ocl = (window as any).OCL;
      if (ocl) resolve(ocl);
      else reject(new Error("ocl_missing"));
    };

    const waitExistingScript = (script: HTMLScriptElement) => {
      let timer: number | null = null;
      const cleanup = () => {
        script.removeEventListener("load", onLoad);
        script.removeEventListener("error", onError);
        if (timer != null) window.clearTimeout(timer);
      };
      const onLoad = () => {
        script.setAttribute("data-ocl-state", "loaded");
        cleanup();
        settleWithWindowOcl();
      };
      const onError = () => {
        script.setAttribute("data-ocl-state", "failed");
        cleanup();
        reject(new Error("ocl_load_failed"));
      };
      script.addEventListener("load", onLoad, { once: true });
      script.addEventListener("error", onError, { once: true });
      timer = window.setTimeout(() => {
        cleanup();
        reject(new Error("ocl_load_timeout"));
      }, OCL_LOAD_TIMEOUT_MS);
    };

    const injectScript = () => {
      const script = document.createElement("script");
      script.src = OCL_CDN;
      script.async = true;
      script.defer = true;
      script.setAttribute("data-ocl", "1");
      script.setAttribute("data-ocl-state", "loading");
      script.onload = () => {
        script.setAttribute("data-ocl-state", "loaded");
        settleWithWindowOcl();
      };
      script.onerror = () => {
        script.setAttribute("data-ocl-state", "failed");
        reject(new Error("ocl_load_failed"));
      };
      document.head.appendChild(script);
    };

    const existing = document.querySelector<HTMLScriptElement>("script[data-ocl='1']");
    if (existing) {
      const state = existing.getAttribute("data-ocl-state");
      if (state === "failed") {
        existing.remove();
        injectScript();
        return;
      }
      waitExistingScript(existing);
      return;
    }
    injectScript();
  });

  oclLoadPromise = oclLoadPromise.catch((err) => {
    lastOclLoadFailureAt = Date.now();
    oclLoadPromise = null;
    throw err;
  });
  return oclLoadPromise;
}

function loadSmilesDrawer(): Promise<any> {
  if (smilesDrawerLoadPromise) return smilesDrawerLoadPromise;
  smilesDrawerLoadPromise = import("smiles-drawer")
    .then((mod: any) => mod?.default ?? mod)
    .catch((err) => {
      smilesDrawerLoadPromise = null;
      throw err;
    });
  return smilesDrawerLoadPromise;
}

async function renderWithSmilesDrawer(
  smiles: string,
  width: number,
  height: number
): Promise<string | null> {
  if (typeof document === "undefined") return null;
  const SD = await loadSmilesDrawer();
  return new Promise((resolve) => {
    const host = document.createElement("div");
    host.style.position = "fixed";
    host.style.left = "-99999px";
    host.style.top = "-99999px";
    host.style.width = "0";
    host.style.height = "0";
    host.style.overflow = "hidden";

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const svgId = `smiles-svg-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    svg.setAttribute("id", svgId);
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    host.appendChild(svg);
    document.body.appendChild(host);

    let settled = false;
    const done = (markup: string | null) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      host.remove();
      resolve(markup ? normalizeSmilesSvgMarkup(markup) : null);
    };
    const timer = window.setTimeout(() => done(null), SMILES_DRAWER_TIMEOUT_MS);

    try {
      const drawer = new SD.SvgDrawer({ width, height });
      SD.parse(
        smiles,
        (tree: any) => {
          try {
            drawer.draw(tree, svgId, "light", false);
            const out = svg.outerHTML;
            done(out && out.includes("<svg") ? out : null);
          } catch {
            done(null);
          }
        },
        () => done(null)
      );
    } catch {
      done(null);
    }
  });
}

function normalizeSmilesSvgMarkup(svg: string): string {
  const style =
    "position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:120%;height:120%;display:block;";
  if (!svg.includes("<svg")) return svg;
  if (svg.includes("data-smiles-svg='1'")) return svg;
  return svg.replace(
    /<svg\b([^>]*)>/i,
    (_m, attrs) => `<svg data-smiles-svg="1" ${attrs} style="${style}">`
  );
}

export async function renderSmilesSvg(
  smiles: string | null | undefined,
  width: number,
  height: number
): Promise<string | null> {
  const s = String(smiles ?? "").trim();
  if (!s) return null;
  const candidates = Array.from(
    new Set(
      [s, s.replace(/:/g, "="), s.replace(/:/g, "")].filter((x) => x.length > 0)
    )
  );
  const cacheKey = `${width}x${height}|${s}`;
  const cached = smilesSvgCache.get(cacheKey);
  if (cached) return cached;
  const existingTask = smilesSvgInflight.get(cacheKey);
  if (existingTask) return existingTask;

  const task = (async () => {
    for (const c of candidates) {
      const svg = await renderWithSmilesDrawer(c, width, height);
      if (svg && svg.includes("<svg")) {
        rememberSvgCache(cacheKey, svg);
        return svg;
      }
    }

    try {
      const OCL = await loadOCL();
      for (const c of candidates) {
        try {
          const mol = OCL.Molecule.fromSmiles(c);
          const svg = mol.toSVG(width, height, "", { suppressCIPParity: true });
          if (svg && svg.includes("<svg")) {
            const out = normalizeSmilesSvgMarkup(svg);
            rememberSvgCache(cacheKey, out);
            return out;
          }
        } catch {
          // try next candidate
        }
      }
    } catch {
      // local renderer already tried above; give up quietly here
    }

    return null;
  })();

  const guardedTask = task.finally(() => {
    smilesSvgInflight.delete(cacheKey);
  });
  smilesSvgInflight.set(cacheKey, guardedTask);
  return guardedTask;
}

export default function SmilesStructure({
  smiles,
  width = 220,
  height = 220,
  className = "",
}: SmilesStructureProps) {
  const renderKey = `${width}x${height}|${String(smiles ?? "").trim()}`;
  const [renderedSvg, setRenderedSvg] = useState<{ key: string; markup: string }>({
    key: "",
    markup: "",
  });

  useEffect(() => {
    const s = String(smiles ?? "").trim();
    if (!s) return;

    let alive = true;
    (async () => {
      const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const svg = await renderSmilesSvg(s, width, height);
          if (!alive) return;
          if (svg) {
            setRenderedSvg({ key: `${width}x${height}|${s}`, markup: svg });
            return;
          }
        } catch {
          // retry
        }
        if (attempt < 2) await wait(80 * (attempt + 1));
      }
      if (alive) setRenderedSvg({ key: `${width}x${height}|${s}`, markup: "" });
    })();

    return () => {
      alive = false;
    };
  }, [smiles, width, height]);

  if (!String(smiles ?? "").trim()) return null;

  const svgMarkup = renderedSvg.key === renderKey ? renderedSvg.markup : "";

  return (
    <>
      <div
        className={`smiles-structure-root ${className}`}
        style={{ width: "100%", height: "100%", background: "white", position: "relative", overflow: "hidden" }}
        dangerouslySetInnerHTML={{ __html: svgMarkup }}
      />
      <style jsx global>{`
        .smiles-structure-root svg {
          width: 100% !important;
          height: 100% !important;
          display: block;
        }
      `}</style>
    </>
  );
}
