"use client";

import { useEffect, useId, useRef, useState } from "react";

type SmilesStructureProps = {
  smiles: string;
  width?: number;
  height?: number;
  className?: string;
};

export default function SmilesStructure({
  smiles,
  width = 180,
  height = 180,
  className = "",
}: SmilesStructureProps) {
  const canvasId = `smiles_canvas_${useId().replace(/:/g, "_")}`;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [renderFailed, setRenderFailed] = useState(false);
  const [remoteFailed, setRemoteFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setRenderFailed(false);

    async function render() {
      const canvas = canvasRef.current;
      if (!canvas || !smiles) {
        setRenderFailed(true);
        return;
      }

      try {
        const mod = await import("smiles-drawer");
        const SmilesDrawer: any = (mod as any).default ?? mod;
        const drawer = new SmilesDrawer.Drawer({
          width,
          height,
          compactDrawing: true,
          terminalCarbons: true,
          explicitHydrogens: false,
        });

        SmilesDrawer.parse(
          smiles,
          (tree: unknown) => {
            if (cancelled) return;
            try {
              drawer.draw(tree, canvasId, "light", false);
            } catch {
              if (!cancelled) setRenderFailed(true);
            }
          },
          () => {
            if (!cancelled) setRenderFailed(true);
          },
        );
      } catch {
        if (!cancelled) setRenderFailed(true);
      }
    }

    void render();
    return () => {
      cancelled = true;
    };
  }, [smiles, width, height, canvasId]);

  if (renderFailed) {
    const remoteUrl = `https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(smiles)}/image`;
    if (!remoteFailed) {
      return (
        <img
          src={remoteUrl}
          alt="substructure"
          className={`w-full h-full object-contain bg-white border border-zinc-200 ${className}`}
          onError={() => setRemoteFailed(true)}
        />
      );
    }
    return (
      <div className={`w-full h-full bg-white border border-zinc-200 p-2 flex items-center justify-center ${className}`}>
        <svg width="86" height="86" viewBox="0 0 86 86" aria-hidden="true" className="text-zinc-400">
          <g fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M30 18h26l13 22-13 22H30L17 40z" />
            <path d="M30 18l13 22-13 22" />
            <path d="M56 18L43 40l13 22" />
            <circle cx="43" cy="40" r="2.5" fill="currentColor" />
          </g>
        </svg>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      id={canvasId}
      width={width}
      height={height}
      className={`w-full h-full bg-white ${className}`}
    />
  );
}
