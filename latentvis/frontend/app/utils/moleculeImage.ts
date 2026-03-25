export type MoleculeImageDataset = "chemprop" | "chemeleon" | "dimenet";

const SLUG_MAX_LENGTH = 72;

export function getBaseRealId(value: string | number | null | undefined): string {
  const raw = String(value ?? "").trim();
  if (raw.startsWith("train::")) return raw.slice("train::".length);
  if (raw.startsWith("test::")) return raw.slice("test::".length);
  if (raw.startsWith("val::")) return raw.slice("val::".length);
  return raw;
}

function slugifyRealId(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return (slug || "molecule").slice(0, SLUG_MAX_LENGTH);
}

function stableHashBase36(value: string): string {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

export function getMoleculeImageGroup(dataset: MoleculeImageDataset): "shared" | "dimenet" {
  return dataset === "dimenet" ? "dimenet" : "shared";
}

export function getMoleculeImageSrc(
  dataset: MoleculeImageDataset,
  realId: string | number | null | undefined
): string | null {
  const baseRealId = getBaseRealId(realId);
  if (!baseRealId) return null;
  const slug = slugifyRealId(baseRealId);
  const hash = stableHashBase36(baseRealId);
  return `/molecule-images/${getMoleculeImageGroup(dataset)}/${slug}-${hash}.png`;
}
