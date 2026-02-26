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

  return oclLoadPromise;
}

export default function SmilesStructure({
  smiles,
  width = 220,
  height = 220,
  className = "",
}: SmilesStructureProps) {
  const [error, setError] = useState<string | null>(null);
  const [svgMarkup, setSvgMarkup] = useState<string>("");

  useEffect(() => {
    const s = String(smiles ?? "").trim();
    if (!s) {
      setError("null");
      setSvgMarkup("");
      return;
    }

    let alive = true;
    (async () => {
      try {
        const OCL = await loadOCL();
        const candidates = Array.from(
          new Set([
            s,
            s.replace(/:/g, "="),
            s.replace(/:/g, ""),
          ].filter((x) => x.length > 0))
        );

        let ok = false;
        let svg = "";
        for (const c of candidates) {
          try {
            const mol = OCL.Molecule.fromSmiles(c);
            svg = mol.toSVG(width, height, "", { suppressCIPParity: true });
            if (svg && svg.includes("<svg")) {
              ok = true;
              break;
            }
          } catch {
            ok = false;
          }
          if (ok) break;
        }

        if (!alive) return;
        setSvgMarkup(ok ? svg : "");
        setError(ok ? null : "null");
      } catch {
        if (!alive) return;
        setSvgMarkup("");
        setError("null");
      }
    })();

    return () => {
      alive = false;
    };
  }, [smiles, width, height]);

  if (error) {
    return (
      <div
        className={`w-full h-full flex items-center justify-center text-sm text-zinc-400 ${className}`}
      >
        null
      </div>
    );
  }

  return (
    <>
      <div
        className={`smiles-structure-root ${className}`}
        style={{ width: "100%", height: "100%", background: "white" }}
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
