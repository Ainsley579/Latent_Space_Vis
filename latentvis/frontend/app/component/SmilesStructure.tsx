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

let oclLoadPromise: Promise<any> | null = null;
let smilesDrawerLoadPromise: Promise<any> | null = null;
const smilesSvgCache = new Map<string, string>();

function loadOCL(): Promise<any> {
  if (typeof window === "undefined") return Promise.reject(new Error("no_window"));
  const w = window as any;
  if (w.OCL) return Promise.resolve(w.OCL);
  if (oclLoadPromise) return oclLoadPromise;

  oclLoadPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>("script[data-ocl='1']");
    if (existing) {
      existing.addEventListener("load", () => resolve((window as any).OCL), { once: true });
      existing.addEventListener("error", () => reject(new Error("ocl_load_failed")), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = OCL_CDN;
    script.async = true;
    script.defer = true;
    script.setAttribute("data-ocl", "1");
    script.onload = () => {
      const ocl = (window as any).OCL;
      if (!ocl) reject(new Error("ocl_missing"));
      else resolve(ocl);
    };
    script.onerror = () => reject(new Error("ocl_load_failed"));
    document.head.appendChild(script);
  });

  oclLoadPromise = oclLoadPromise.catch((err) => {
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

    const done = (markup: string | null) => {
      host.remove();
      resolve(markup ? normalizeSmilesSvgMarkup(markup) : null);
    };

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

  try {
    const OCL = await loadOCL();
    for (const c of candidates) {
      try {
        const mol = OCL.Molecule.fromSmiles(c);
        const svg = mol.toSVG(width, height, "", { suppressCIPParity: true });
        if (svg && svg.includes("<svg")) return normalizeSmilesSvgMarkup(svg);
      } catch {
        // try next candidate
      }
    }
  } catch {
    // fallback to smiles-drawer below
  }

  for (const c of candidates) {
    const svg = await renderWithSmilesDrawer(c, width, height);
    if (svg && svg.includes("<svg")) return svg;
  }
  return null;
}

export default function SmilesStructure({
  smiles,
  width = 220,
  height = 220,
  className = "",
}: SmilesStructureProps) {
  const [svgMarkup, setSvgMarkup] = useState<string>("");

  useEffect(() => {
    const s = String(smiles ?? "").trim();
    if (!s) return;

    let alive = true;
    (async () => {
      const cacheKey = `${s}__${width}x${height}`;
      const cached = smilesSvgCache.get(cacheKey);
      if (cached && alive) setSvgMarkup(cached);

      const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const svg = await renderSmilesSvg(s, width, height);
          if (!alive) return;
          if (svg) {
            smilesSvgCache.set(cacheKey, svg);
            setSvgMarkup(svg);
            return;
          }
        } catch {
          // retry
        }
        if (attempt < 2) await wait(80 * (attempt + 1));
      }
    })();

    return () => {
      alive = false;
    };
  }, [smiles, width, height]);

  if (!String(smiles ?? "").trim()) return null;

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
