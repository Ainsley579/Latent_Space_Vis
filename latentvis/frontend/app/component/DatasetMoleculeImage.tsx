"use client";

import Image from "next/image";
import { useMemo, useState } from "react";
import SmilesStructure from "./SmilesStructure";
import { getBaseRealId, getMoleculeImageSrc, type MoleculeImageDataset } from "../utils/moleculeImage";

type DatasetMoleculeImageProps = {
  dataset: MoleculeImageDataset;
  realId: string | number | null | undefined;
  smiles: string | null | undefined;
  className?: string;
};

export default function DatasetMoleculeImage({
  dataset,
  realId,
  smiles,
  className = "",
}: DatasetMoleculeImageProps) {
  const [imageFailed, setImageFailed] = useState(false);
  const src = useMemo(() => getMoleculeImageSrc(dataset, realId), [dataset, realId]);
  const text = String(smiles ?? "").trim();
  const showSvg = !!text && (imageFailed || !src);
  const showImage = !imageFailed && !!src;
  const alt = getBaseRealId(realId) || text || "molecule";

  return (
    <div className={`relative w-full h-full overflow-hidden bg-white ${className}`}>
      {showImage && (
        <Image
          src={src}
          alt={alt}
          fill
          unoptimized
          sizes="220px"
          className="object-contain"
          draggable={false}
          onError={() => setImageFailed(true)}
        />
      )}
      {showSvg && (
        <div className="absolute inset-0">
          <SmilesStructure smiles={text} className="w-full h-full" background="transparent" />
        </div>
      )}
      {!showImage && !showSvg && (
        <div className="w-full h-full flex items-center justify-center text-sm text-zinc-400">null</div>
      )}
    </div>
  );
}
