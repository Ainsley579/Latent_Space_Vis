"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { CLUSTER_COLORS, HIGHLIGHT_RED } from "../theme/colors";
import { fetchJsonCached } from "../utils/fetchCache";
import { renderSmilesSvg } from "./SmilesStructure";

type ScatterPoint = {
  real_id: string;
  x: number;
  y: number;
  cluster_id: number;
  split?: "train" | "test" | "val";
  y_true?: number;
  y_pred?: number;
};

type SplitTag = "train" | "test" | "val";
type SplitMetaTag = SplitTag | "both";

type ColorMode = "cluster" | "label" | "split" | "y_true" | "y_pred" | "y_abs_err";

function normalizeLayerKey(value: unknown): string {
  const raw = String(value ?? "").trim();
  if (raw === "fp") return "fingerprint";
  if (/^fnn_layer_/.test(raw)) return raw.replace(/^fnn_layer_/, "ffn_layer_");
  return raw;
}

function stripSplitTag(rawId: string): string {
  const raw = String(rawId ?? "").trim();
  if (!raw) return "";
  if (raw.startsWith("train::")) return raw.slice("train::".length).trim();
  if (raw.startsWith("test::")) return raw.slice("test::".length).trim();
  if (raw.startsWith("val::")) return raw.slice("val::".length).trim();
  return raw;
}

function normalizeSmilesKey(raw: string): string {
  return String(raw ?? "").trim().replace(/\s+/g, "");
}

type ScatterPlotProps = {
  compactOverview?: any;
  dataPathXY?: string;
  dataPathCluster?: string;
  secondaryDataPathXY?: string;
  secondaryDataPathCluster?: string;
  splitMapPath?: string;
  splitMapDataset?: string;
  dataSplit?: "all" | "train" | "test";
  yTrueLookup?: Record<string, number>;
  yPredLookup?: Record<string, number>;
  projection?: "pca" | "mds" | "umap" | "tsne";
  onProjectionChange?: (next: "pca" | "mds" | "umap" | "tsne") => void;
  activeCluster?: number | null;
  activeIds?: string[] | null;
  highlightId?: string | null;
  modelKey?: string;
  layerKey?: string;
  onPointSelect?: (sel: { model?: string; layer?: string; cluster: number; real_id: string }) => void;
  onPointHoverIdChange?: (id: string | null) => void;
  onClusterSelect?: (sel: { model?: string; layer?: string; cluster: number; real_ids: string[] }) => void;
  onHighlightIdsSelect?: (ids: string[]) => void;
  onClearSelection?: () => void;
  getSmilesById?: (realId: string | number) => string | null;
  onUserClusteringUpload?: (
    payload: {
      fileName: string;
      labelNames: string[];
      rows: Array<{ real_id: string; user_cluster_id: number }>;
    } | null
  ) => void;
};

export default function ScatterPlot({
  compactOverview,
  dataPathXY,
  dataPathCluster,
  secondaryDataPathXY,
  secondaryDataPathCluster,
  splitMapPath,
  splitMapDataset,
  dataSplit,
  yTrueLookup,
  yPredLookup,
  projection,
  onProjectionChange,
  activeCluster,
  activeIds,
  highlightId,
  modelKey,
  layerKey,
  onPointSelect,
  onPointHoverIdChange,
  onClusterSelect,
  onHighlightIdsSelect,
  onClearSelection,
  getSmilesById,
  onUserClusteringUpload,
}: ScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const rowsRef = useRef<ScatterPoint[] | null>(null);
  const onPointSelectRef = useRef(onPointSelect);
  const onPointHoverIdChangeRef = useRef(onPointHoverIdChange);
  const onClearSelectionRef = useRef(onClearSelection);
  const getSmilesByIdRef = useRef(getSmilesById);
  const modelKeyRef = useRef(modelKey);
  const layerKeyRef = useRef(layerKey);
  const yTrueByIdRef = useRef<Record<string, number>>({});
  const yPredByIdRef = useRef<Record<string, number>>({});
  const yAbsErrByIdRef = useRef<Record<string, number>>({});
  const splitByIdRef = useRef<Record<string, SplitMetaTag>>({});
  const [legendClusters, setLegendClusters] = useState<number[]>([]);
  const [legendClusterIds, setLegendClusterIds] = useState<Record<number, string[]>>({});
  const [colorMode, setColorMode] = useState<ColorMode>("cluster");
  const [labelById, setLabelById] = useState<Record<string, string>>({});
  const [labelBySmiles, setLabelBySmiles] = useState<Record<string, string>>({});
  const [labelColors, setLabelColors] = useState<Record<string, string>>({});
  const [labelLegend, setLabelLegend] = useState<string[]>([]);
  const [labelFileName, setLabelFileName] = useState<string>("");
  const [labelLoadError, setLabelLoadError] = useState<string | null>(null);
  const [labelMatchInfo, setLabelMatchInfo] = useState<string>("");
  const [hasSplitData, setHasSplitData] = useState(false);
  const [visibleSplits, setVisibleSplits] = useState<SplitTag[]>([]);
  const [yTrueRange, setYTrueRange] = useState<{ min: number; max: number } | null>(null);
  const [yPredRange, setYPredRange] = useState<{ min: number; max: number } | null>(null);
  const [yAbsErrRange, setYAbsErrRange] = useState<{ min: number; max: number } | null>(null);
  const [ySharedRange, setYSharedRange] = useState<{ min: number; max: number } | null>(null);
  const colorModeRef = useRef<ColorMode>("cluster");

  function idLookupKeys(raw: string): string[] {
    const s = String(raw ?? "").trim();
    if (!s) return [];
    const keys = new Set<string>([s]);
    keys.add(s.toLowerCase());
    const stripped = (() => {
      const idx = s.indexOf("::");
      return idx >= 0 ? s.slice(idx + 2) : s;
    })();
    keys.add(stripped);
    keys.add(stripped.toLowerCase());
    if (/^-?\d+(\.0+)?$/.test(stripped)) {
      const n = Number(stripped);
      if (Number.isFinite(n)) {
        keys.add(String(Math.trunc(n)));
        keys.add(String(n));
      }
    }
    const parsed = Number(stripped);
    if (Number.isFinite(parsed)) {
      keys.add(String(parsed));
      if (Math.abs(parsed - Math.round(parsed)) < 1e-9) {
        keys.add(String(Math.round(parsed)));
      }
    }
    return Array.from(keys);
  }

  function lookupLabelById(map: Record<string, string>, rawId: string): string | undefined {
    const keys = idLookupKeys(rawId);
    for (let i = 0; i < keys.length; i += 1) {
      const hit = map[keys[i]];
      if (hit) return hit;
    }
    return undefined;
  }

  function buildLabelColors(labels: string[]): Record<string, string> {
    const uniq = Array.from(new Set(labels.map((x) => String(x).trim()).filter((x) => x.length > 0))).sort((a, b) =>
      a.localeCompare(b)
    );
    const out: Record<string, string> = {};
    const total = Math.max(1, uniq.length);
    uniq.forEach((label, idx) => {
      out[label] = d3.interpolateTurbo((idx + 0.5) / total);
    });
    return out;
  }

  function resolveBaseFillByMode(mode: ColorMode, id: string, cluster: number, split?: SplitTag): string {
    if (mode === "split") {
      if (split === "val") return "#1e3a8a";
      if (split === "test") return "#2563eb";
      if (split === "train") return "#f59e0b";
      return "#9ca3af";
    }
    if (mode === "label") {
      const direct = lookupLabelById(labelById, id);
      if (direct) return labelColors[direct] ?? "#9ca3af";
      const smiles = getSmilesByIdRef.current?.(String(id));
      if (smiles) {
        const bySmiles = labelBySmiles[normalizeSmilesKey(String(smiles))];
        if (bySmiles) return labelColors[bySmiles] ?? "#9ca3af";
      }
      return "#9ca3af";
    }
    if (mode === "y_true") {
      const v = Number(yTrueByIdRef.current[id]);
      if (!Number.isFinite(v) || !ySharedRange) return "#9ca3af";
      const { min, max } = ySharedRange;
      const t = max > min ? (v - min) / (max - min) : 0.5;
      return d3.interpolateYlGnBu(Math.max(0, Math.min(1, t)));
    }
    if (mode === "y_pred") {
      const v = Number(yPredByIdRef.current[id]);
      if (!Number.isFinite(v) || !ySharedRange) return "#9ca3af";
      const { min, max } = ySharedRange;
      const t = max > min ? (v - min) / (max - min) : 0.5;
      return d3.interpolateYlGnBu(Math.max(0, Math.min(1, t)));
    }
    if (mode === "y_abs_err") {
      const v = Number(yAbsErrByIdRef.current[id]);
      if (!Number.isFinite(v) || !yAbsErrRange) return "#9ca3af";
      const { min, max } = yAbsErrRange;
      const t = max > min ? (v - min) / (max - min) : 0.5;
      return d3.interpolateYlOrRd(Math.max(0, Math.min(1, t)));
    }
    return CLUSTER_COLORS[Math.max(0, cluster) % CLUSTER_COLORS.length];
  }

  function resolveBaseFill(id: string, cluster: number, split?: SplitTag): string {
    return resolveBaseFillByMode(colorMode, id, cluster, split);
  }

  function repaintBaseFillForMode(mode: ColorMode) {
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;
    const circles = Array.from(svg.querySelectorAll("circle[data-id]")) as SVGCircleElement[];
    circles.forEach((el) => {
      const id = el.getAttribute("data-id") || "";
      const cluster = Number(el.getAttribute("data-cluster") || "0");
      const split = (el.getAttribute("data-split") || "") as SplitTag | "";
      const fill = resolveBaseFillByMode(mode, id, cluster, split || undefined);
      el.setAttribute("data-base-fill", fill);
      if (!highlightId || id !== String(highlightId)) {
        el.setAttribute("fill", fill);
      }
    });
  }

  function applySelectionStateToSvg(svg: SVGElement) {
    const rows = rowsRef.current;
    if (!rows) return;
    const activeIdSet = activeIds ? new Set(activeIds.map((x) => String(x))) : null;
    const activeLookup = new Set<string>();
    if (activeIdSet) {
      activeIdSet.forEach((id) => {
        idLookupKeys(id).forEach((k) => activeLookup.add(k));
      });
    }
    const highlightLookup = highlightId ? new Set(idLookupKeys(String(highlightId))) : null;
    const circles = Array.from(svg.querySelectorAll("circle[data-id]")) as SVGCircleElement[];
    let matchedActiveCount = 0;
    const activeMatchedByCircle = new Map<string, boolean>();
    if (activeLookup.size > 0) {
      circles.forEach((el) => {
        const id = el.getAttribute("data-id") || "";
        const hit = idLookupKeys(id).some((k) => activeLookup.has(k));
        if (hit) matchedActiveCount += 1;
        activeMatchedByCircle.set(id, hit);
      });
    }
    const hasUsableActiveSelection = activeLookup.size > 0 && matchedActiveCount > 0;
    circles.forEach((el) => {
      const id = el.getAttribute("data-id") || "";
      const cluster = Number(el.getAttribute("data-cluster") || "0");
      const split = (el.getAttribute("data-split") || "") as SplitTag | "";
      const computedBaseFill = resolveBaseFill(id, cluster, split || undefined);
      el.setAttribute("data-base-fill", computedBaseFill);
      const baseFill = computedBaseFill;
      const isHovered = !!highlightLookup && idLookupKeys(id).some((k) => highlightLookup.has(k));
      const inActiveIds = activeMatchedByCircle.get(id) === true;

      el.setAttribute("r", isHovered ? "5.2" : "3.2");
      el.setAttribute("fill", isHovered ? "#f59e0b" : baseFill);
      if (isHovered) {
        el.setAttribute("fill-opacity", "1");
        el.setAttribute("stroke", "#dc2626");
        el.setAttribute("stroke-width", "1.8");
      } else if (hasUsableActiveSelection) {
        el.setAttribute("fill-opacity", inActiveIds ? "1" : "0.22");
        el.setAttribute("stroke", inActiveIds ? "#dc2626" : "none");
        el.setAttribute("stroke-width", "0.9");
      } else if (activeCluster == null) {
        el.setAttribute("fill-opacity", "0.75");
        el.setAttribute("stroke", "none");
        el.setAttribute("stroke-width", "0.9");
      } else {
        const match = cluster === activeCluster;
        el.setAttribute("fill-opacity", match ? "1" : "0.22");
        el.setAttribute("stroke", match ? "#dc2626" : "none");
        el.setAttribute("stroke-width", "0.9");
      }
    });

    d3.select(svg).selectAll("path.hover-star-overlay-path").remove();
    if (highlightId) {
      const targetId = String(highlightId);
      const hitKeys = new Set(idLookupKeys(targetId));
      const hitCircle = circles.find((c) => {
        const cid = c.getAttribute("data-id") || "";
        return idLookupKeys(cid).some((k) => hitKeys.has(k));
      });
      if (hitCircle && hitCircle.parentElement) {
        const cx = Number(hitCircle.getAttribute("cx") || "0");
        const cy = Number(hitCircle.getAttribute("cy") || "0");
        const fillColor =
          hitCircle.getAttribute("data-base-fill") ||
          CLUSTER_COLORS[Math.max(0, Number(hitCircle.getAttribute("data-cluster") || "0")) % CLUSTER_COLORS.length];
        const star = d3.symbol().type(d3.symbolStar).size(180);
        d3.select(hitCircle.parentElement)
          .append("path")
          .attr("class", "hover-star-overlay-path")
          .attr("d", star)
          .attr("transform", `translate(${cx},${cy})`)
          .attr("fill", fillColor)
          .attr("stroke", HIGHLIGHT_RED)
          .attr("stroke-width", 1.2)
          .attr("opacity", 0.95)
          .style("pointer-events", "none");
      }
    }
  }

  function setColorModeAndRefresh(next: ColorMode) {
    colorModeRef.current = next;
    setColorMode(next);
    // Repaint twice to avoid first-click race with async draw/effect passes.
    requestAnimationFrame(() => repaintBaseFillForMode(next));
    setTimeout(() => repaintBaseFillForMode(next), 0);
  }

  function emitHighlightIds(ids: string[]) {
    const uniq = Array.from(new Set(ids.map((x) => String(x)).filter((x) => x.length > 0)));
    onHighlightIdsSelect?.(uniq);
  }

  function idsForSplit(target: SplitTag): string[] {
    const rows = rowsRef.current ?? [];
    return rows.filter((d) => d.split === target).map((d) => String(d.real_id));
  }

  function idsForLabel(targetLabel: string): string[] {
    const rows = rowsRef.current ?? [];
        return rows
      .filter((d) => {
        const rid = String(d.real_id);
        const direct = lookupLabelById(labelById, rid);
        if (direct) return direct === targetLabel;
        const smiles = getSmilesByIdRef.current?.(rid);
        if (!smiles) return false;
        return (labelBySmiles[normalizeSmilesKey(String(smiles))] ?? "") === targetLabel;
      })
      .map((d) => String(d.real_id));
  }

  function csvEscape(val: string): string {
    const s = String(val ?? "");
    if (s.includes(",") || s.includes('"') || s.includes("\n") || s.includes("\r")) {
      return `"${s.replace(/"/g, '""')}"`;
    }
    return s;
  }

  function downloadDatasetCsv() {
    const splitMap = splitByIdRef.current ?? {};
    const lines = ["real_id,smiles,split"];
    const splitIds = Object.keys(splitMap);
    if (splitIds.length > 0) {
      splitIds.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      splitIds.forEach((rid) => {
        const split = String(splitMap[rid] ?? "").trim();
        const smiles = String(getSmilesByIdRef.current?.(rid) ?? "");
        lines.push([csvEscape(rid), csvEscape(smiles), csvEscape(split)].join(","));
      });
    } else {
      const rows = rowsRef.current ?? [];
      if (!rows.length) return;
      const seen = new Set<string>();
      rows.forEach((d) => {
        const rid = String(d.real_id ?? "").trim();
        if (!rid) return;
        const split = String(d.split ?? "").trim();
        const key = `${rid}__${split}`;
        if (seen.has(key)) return;
        seen.add(key);
        const smiles = String(getSmilesByIdRef.current?.(rid) ?? "");
        lines.push([csvEscape(rid), csvEscape(smiles), csvEscape(split)].join(","));
      });
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const datasetName = splitMapDataset ? String(splitMapDataset).trim() : "dataset";
    a.href = url;
    a.download = `${datasetName}_scatter_export.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  useEffect(() => {
    colorModeRef.current = colorMode;
  }, [colorMode]);

  useEffect(() => {
    repaintBaseFillForMode(colorMode);
  }, [colorMode, ySharedRange]);

  function handleLabelFileUpload(file: File) {
    setLabelLoadError(null);
    setLabelMatchInfo("");
    file
      .text()
      .then((text) => {
        const parseCandidates = [
          d3.csvParse(text),
          d3.tsvParse(text),
          d3.dsvFormat(";").parse(text),
        ];
        let rows = parseCandidates[0];
        let bestScore = -1;
        parseCandidates.forEach((candidate) => {
          const colCount = Array.isArray(candidate?.columns) ? candidate.columns.length : 0;
          const rowCount = Array.isArray(candidate) ? candidate.length : 0;
          const score = colCount * 100000 + rowCount;
          if (score > bestScore) {
            bestScore = score;
            rows = candidate;
          }
        });
        if (!rows.length) throw new Error("CSV is empty.");
        const columns = rows.columns ?? [];

        const normalizeColKey = (s: string) => String(s ?? "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "_");
        const columnByNormalized = new Map<string, string>();
        columns.forEach((c) => columnByNormalized.set(normalizeColKey(c), c));
        const findCol = (aliases: string[]): string | undefined => {
          for (let i = 0; i < aliases.length; i += 1) {
            const hit = columnByNormalized.get(aliases[i]);
            if (hit) return hit;
          }
          return undefined;
        };

        const labelCol =
          findCol([
            "label",
            "labelled",
            "labeled",
            "user_label",
            "user_cluster",
            "user_cluster_id",
            "cluster",
            "cluster_id",
            "group",
            "class",
          ]) ??
          columns.find((c) => {
            const k = normalizeColKey(c);
            return k.includes("label") || k.includes("cluster") || k.includes("group") || k.includes("class");
          });
        if (!labelCol) {
          throw new Error("CSV must contain a label column (e.g. label / labelled / user_cluster_id).");
        }

        let idCol =
          findCol([
            "real_id",
            "id",
            "realid",
            "mol_id",
            "molid",
            "molecule_id",
            "molecule_idx",
            "mol_index",
            "compound_id",
            "sample_id",
            "data_id",
            "datapoint_id",
            "idx",
            "index",
          ]) ??
          columns.find((c) => {
            if (c === labelCol) return false;
            const k = normalizeColKey(c);
            if (!k.endsWith("_id")) return false;
            if (k.includes("cluster") || k.includes("label")) return false;
            return true;
          });
        const smilesCol = findCol(["smiles", "smile"]);
        if (!idCol) {
          const points = rowsRef.current ?? [];
          const pointKeySet = new Set<string>();
          points.forEach((pt) => {
            idLookupKeys(String(pt.real_id)).forEach((k) => pointKeySet.add(k));
            const stripped = stripSplitTag(String(pt.real_id));
            idLookupKeys(stripped).forEach((k) => pointKeySet.add(k));
          });
          let bestCol: string | null = null;
          let bestMatch = 0;
          columns.forEach((col) => {
            if (col === labelCol || col === smilesCol) return;
            let matched = 0;
            let total = 0;
            rows.forEach((row) => {
              const value = String((row as Record<string, unknown>)[col] ?? "").trim();
              if (!value) return;
              total += 1;
              const keys = idLookupKeys(value);
              if (keys.some((k) => pointKeySet.has(k))) matched += 1;
            });
            const ratio = total > 0 ? matched / total : 0;
            if (ratio > bestMatch) {
              bestMatch = ratio;
              bestCol = col;
            }
          });
          if (bestCol && bestMatch >= 0.2) idCol = bestCol;
        }

        const nextLabelById: Record<string, string> = {};
        const nextLabelBySmiles: Record<string, string> = {};
        const uploadedIdLabelRows: Array<{ real_id: string; label: string }> = [];
        const points = rowsRef.current ?? [];
        const countMatchedScatterPoints = (
          mapById: Record<string, string>,
          mapBySmiles: Record<string, string>
        ): number => {
          let matched = 0;
          points.forEach((pt) => {
            const rid = String(pt.real_id);
            if (lookupLabelById(mapById, rid)) {
              matched += 1;
              return;
            }
            const smiles = getSmilesByIdRef.current?.(rid);
            if (smiles && mapBySmiles[normalizeSmilesKey(String(smiles))]) {
              matched += 1;
            }
          });
          return matched;
        };
        if (smilesCol) {
          rows.forEach((row) => {
            const smiles = String((row as Record<string, unknown>)[smilesCol] ?? "").trim();
            const label = String((row as Record<string, unknown>)[labelCol] ?? "").trim();
            if (!smiles || !label) return;
            nextLabelBySmiles[normalizeSmilesKey(smiles)] = label;
          });
        }
        if (idCol) {
          rows.forEach((row) => {
            const id = String((row as Record<string, unknown>)[idCol] ?? "").trim();
            const label = String((row as Record<string, unknown>)[labelCol] ?? "").trim();
            if (!id || !label) return;
            uploadedIdLabelRows.push({ real_id: id, label });
            idLookupKeys(id).forEach((k) => {
              nextLabelById[k] = label;
            });
            const canonical = stripSplitTag(id);
            if (canonical) {
              idLookupKeys(canonical).forEach((k) => {
                nextLabelById[k] = label;
              });
            }
          });
        } else if (!smilesCol) {
          if (points.length !== rows.length) {
            throw new Error("Missing id column. Please provide 'real_id' or 'id'.");
          }
          rows.forEach((row, idx) => {
            const id = String(points[idx]?.real_id ?? "").trim();
            const label = String((row as Record<string, unknown>)[labelCol] ?? "").trim();
            if (!id || !label) return;
            idLookupKeys(id).forEach((k) => {
              nextLabelById[k] = label;
            });
          });
        }
        let matchedForColorPre = countMatchedScatterPoints(nextLabelById, nextLabelBySmiles);
        if (matchedForColorPre === 0 && points.length > 0 && points.length === rows.length) {
          rows.forEach((row, idx) => {
            const ridRaw = String(points[idx]?.real_id ?? "").trim();
            const label = String((row as Record<string, unknown>)[labelCol] ?? "").trim();
            if (!ridRaw || !label) return;
            idLookupKeys(ridRaw).forEach((k) => {
              nextLabelById[k] = label;
            });
            const canonical = stripSplitTag(ridRaw);
            if (canonical) {
              idLookupKeys(canonical).forEach((k) => {
                nextLabelById[k] = label;
              });
              uploadedIdLabelRows.push({ real_id: canonical, label });
            }
          });
          matchedForColorPre = countMatchedScatterPoints(nextLabelById, nextLabelBySmiles);
        }

        const labels = Array.from(new Set([...Object.values(nextLabelById), ...Object.values(nextLabelBySmiles)]));
        if (!labels.length) throw new Error("No valid id-label rows found.");
        const legend = labels.sort((a, b) => a.localeCompare(b));
        const colors = buildLabelColors(legend);
        const labelToCluster = new Map<string, number>();
        legend.forEach((name, idx) => labelToCluster.set(name, idx));
        const userRowMap = new Map<string, number>();
        if (uploadedIdLabelRows.length > 0) {
          uploadedIdLabelRows.forEach(({ real_id, label }) => {
            const cid = labelToCluster.get(label);
            if (cid == null) return;
            const rid = stripSplitTag(real_id);
            if (!rid) return;
            userRowMap.set(rid, cid);
          });
        } else {
          points.forEach((pt) => {
            const rid = String(pt.real_id);
            let label = lookupLabelById(nextLabelById, rid);
            if (!label) {
              const smiles = getSmilesByIdRef.current?.(rid);
              if (smiles) label = nextLabelBySmiles[normalizeSmilesKey(String(smiles))];
            }
            if (!label) return;
            const cid = labelToCluster.get(label);
            if (cid == null) return;
            const canonicalId = stripSplitTag(rid);
            if (!canonicalId) return;
            userRowMap.set(canonicalId, cid);
          });
        }
        const userRows: Array<{ real_id: string; user_cluster_id: number }> = [];
        userRowMap.forEach((user_cluster_id, real_id) => {
          userRows.push({ real_id, user_cluster_id });
        });
        userRows.sort((a, b) => a.real_id.localeCompare(b.real_id, undefined, { numeric: true }));
        const scatterRows = rowsRef.current ?? [];
        let matchedForColor = 0;
        scatterRows.forEach((pt) => {
          const rid = String(pt.real_id);
          const labelByIdHit = lookupLabelById(nextLabelById, rid);
          if (labelByIdHit) {
            matchedForColor += 1;
            return;
          }
          const smiles = getSmilesByIdRef.current?.(rid);
          if (smiles && nextLabelBySmiles[normalizeSmilesKey(String(smiles))]) {
            matchedForColor += 1;
          }
        });
        setLabelById(nextLabelById);
        setLabelBySmiles(nextLabelBySmiles);
        setLabelColors(colors);
        setLabelLegend(legend);
        setLabelFileName(file.name);
        setColorModeAndRefresh("label");
        setLabelMatchInfo(`matched ${matchedForColor} / ${scatterRows.length} points`);
        if (scatterRows.length > 0 && matchedForColor === 0) {
          setLabelLoadError("Upload succeeded, but 0 scatter points matched your id/smiles columns.");
        }
        if (userRows.length === 0) {
          setLabelLoadError("Labels loaded, but no points matched current scatter IDs/SMILES.");
          onUserClusteringUpload?.(null);
          return;
        }
        onUserClusteringUpload?.({
          fileName: file.name,
          labelNames: legend,
          rows: userRows,
        });
      })
      .catch((err) => {
        setLabelLoadError(err instanceof Error ? err.message : String(err));
        setLabelMatchInfo("");
        onUserClusteringUpload?.(null);
      });
  }

  useEffect(() => {
    onPointSelectRef.current = onPointSelect;
  }, [onPointSelect]);

  useEffect(() => {
    onPointHoverIdChangeRef.current = onPointHoverIdChange;
  }, [onPointHoverIdChange]);

  useEffect(() => {
    onClearSelectionRef.current = onClearSelection;
  }, [onClearSelection]);

  useEffect(() => {
    getSmilesByIdRef.current = getSmilesById;
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;
    requestAnimationFrame(() => {
      repaintBaseFillForMode(colorModeRef.current);
      applySelectionStateToSvg(svg);
    });
  }, [getSmilesById]);

  useEffect(() => {
    modelKeyRef.current = modelKey;
  }, [modelKey]);

  useEffect(() => {
    layerKeyRef.current = layerKey;
  }, [layerKey]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";
    container.style.position = "relative";

    const ro = new ResizeObserver(() => draw());
    ro.observe(container);

    let lastRows: ScatterPoint[] | null = null;
    const tooltip = d3
      .select(document.body)
      .append("div")
      .style("position", "fixed")
      .style("display", "none")
      .style("pointer-events", "none")
      .style("background", "rgba(255,255,255,0.95)")
      .style("border", "1px solid #d1d5db")
      .style("box-shadow", "0 6px 20px rgba(0,0,0,0.12)")
      .style("padding", "6px")
      .style("border-radius", "6px")
      .style("z-index", "9999");

    const tooltipSvgWrap = tooltip
      .append("div")
      .style("display", "block")
      .style("width", "300px")
      .style("height", "300px")
      .style("background", "white")
      .style("position", "relative")
      .style("overflow", "hidden");
    const tooltipMeta = tooltip
      .append("div")
      .style("margin-top", "6px")
      .style("font-size", "11px")
      .style("line-height", "1.3")
      .style("color", "#374151");
    let hoverToken = 0;

    function draw() {
      if (!container || !lastRows) return;
      container.innerHTML = "";
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);

      const margin = { top: 8, right: 8, bottom: 8, left: 8 };
      const innerW = Math.max(1, width - margin.left - margin.right);
      const innerH = Math.max(1, height - margin.top - margin.bottom);

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
      const gRoot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
      const g = gRoot.append("g");

      const xExtent = d3.extent(lastRows, (d) => d.x) as [number, number];
      const yExtent = d3.extent(lastRows, (d) => d.y) as [number, number];
      const x = d3.scaleLinear().domain(xExtent).range([0, innerW]).nice();
      const y = d3.scaleLinear().domain(yExtent).range([innerH, 0]).nice();

      g.append("g")
        .selectAll("circle")
        .data(lastRows)
        .enter()
        .append("circle")
        .attr("data-id", (d) => String(d.real_id))
        .attr("data-cluster", (d) => String(d.cluster_id))
        .attr("data-split", (d) => String(d.split ?? ""))
        .attr("data-y-true", (d) => (Number.isFinite(Number(d.y_true)) ? String(d.y_true) : ""))
        .attr("data-y-pred", (d) => (Number.isFinite(Number(d.y_pred)) ? String(d.y_pred) : ""))
        .attr("data-base-fill", (d) =>
          resolveBaseFillByMode(colorModeRef.current, String(d.real_id), d.cluster_id, d.split)
        )
        .attr("cx", (d) => x(d.x))
        .attr("cy", (d) => y(d.y))
        .attr("r", 3.2)
        .attr("fill", (d) =>
          resolveBaseFillByMode(colorModeRef.current, String(d.real_id), d.cluster_id, d.split)
        )
        .attr("fill-opacity", 0.75)
        .attr("stroke", "none")
        .attr("stroke-width", 0.9)
        .style("cursor", "default")
        .on("mouseover", (event, d) => {
          const token = ++hoverToken;
          const fmtNum = (v: unknown, digits = 5): string => {
            const n = Number(v);
            if (!Number.isFinite(n)) return "N/A";
            return n.toFixed(digits);
          };
          const yt = Number(d.y_true);
          const yp = Number(d.y_pred);
          const absErr = Number.isFinite(yt) && Number.isFinite(yp) ? Math.abs(yt - yp) : NaN;
          const logDiff = Number.isFinite(absErr) ? Math.log(absErr + 1e-6) : NaN;
          const splitText = d.split ? String(d.split) : "N/A";
          tooltipMeta.html(
            [
              `<div><b>ID:</b> ${String(d.real_id)}</div>`,
              `<div><b>cluster_id:</b> ${d.cluster_id}</div>`,
              `<div><b>split:</b> ${splitText}</div>`,
              `<div><b>x:</b> ${fmtNum(d.x)} <b>y:</b> ${fmtNum(d.y)}</div>`,
              `<div><b>y_true:</b> ${fmtNum(yt)}</div>`,
              `<div><b>y_pred:</b> ${fmtNum(yp)}</div>`,
              `<div><b>|y_true-y_pred|:</b> ${fmtNum(absErr)}</div>`,
              `<div><b>ln_diff:</b> ${fmtNum(logDiff)}</div>`,
            ].join("")
          );
          const smiles = getSmilesByIdRef.current?.(String(d.real_id)) ?? null;
          onPointHoverIdChangeRef.current?.(String(d.real_id));
          tooltipSvgWrap.html(`<div style="width:300px;height:300px;display:flex;align-items:center;justify-content:center;color:#a1a1aa;font-size:14px;">loading</div>`);
          if (smiles) {
            void renderSmilesSvg(smiles, 300, 300).then((svg) => {
              if (token !== hoverToken) return;
              if (svg) {
                tooltipSvgWrap.html(svg);
              } else {
                tooltipSvgWrap.html(`<div style="width:300px;height:300px;display:flex;align-items:center;justify-content:center;color:#a1a1aa;font-size:14px;">null</div>`);
              }
            }).catch(() => {
              if (token !== hoverToken) return;
              tooltipSvgWrap.html(`<div style="width:300px;height:300px;display:flex;align-items:center;justify-content:center;color:#a1a1aa;font-size:14px;">null</div>`);
            });
          } else {
            tooltipSvgWrap.html(`<div style="width:300px;height:300px;display:flex;align-items:center;justify-content:center;color:#a1a1aa;font-size:14px;">null</div>`);
          }
          tooltip.style("display", "block");
          positionTooltip(event.clientX, event.clientY);
        })
        .on("mousemove", (event) => {
          positionTooltip(event.clientX, event.clientY);
        })
        .on("mouseout", () => {
          hoverToken += 1;
          onPointHoverIdChangeRef.current?.(null);
          tooltip.style("display", "none");
        })
        .on("click", (event) => {
          event.stopPropagation();
        });

      // Hover star is updated in a separate effect and rendered inside the same circle group.

      const extent: [[number, number], [number, number]] = [
        [margin.left, margin.top],
        [margin.left + innerW, margin.top + innerH],
      ];
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.01, 8])
        .translateExtent(extent)
        .extent(extent)
        .on("zoom", (event) => {
          const t = event.transform;
          g.attr("transform", `translate(${t.x - margin.left},${t.y - margin.top}) scale(${t.k})`);
        });

      svg.call(zoom);
      svg.call(zoom.transform, d3.zoomIdentity.translate(margin.left, margin.top));

      svg.on("click", () => {
        onClearSelectionRef.current?.();
      });

      const svgEl = svg.node();
      if (svgEl) applySelectionStateToSvg(svgEl);

      function positionTooltip(clientX: number, clientY: number) {
        const pad = 12;
        const tipBounds = (tooltip.node() as HTMLDivElement | null)?.getBoundingClientRect();
        const tipW = tipBounds?.width ?? 0;
        const tipH = tipBounds?.height ?? 0;

        let x = clientX + pad;
        let y = clientY + pad;

        if (x + tipW > window.innerWidth) {
          x = Math.max(pad, clientX - pad - tipW);
        }
        if (y + tipH > window.innerHeight) {
          y = Math.max(pad, clientY - pad - tipH);
        }

        tooltip.style("left", `${x}px`).style("top", `${y}px`);
      }
    }

    const fetchScatterJson = async (url: string, kind: "xy" | "cluster") => {
      try {
        return await fetchJsonCached(url);
      } catch {
        throw new Error(`Failed to fetch scatter ${kind} data`);
      }
    };

    const fetchSplitMap = async () => {
      if (!splitMapPath || !splitMapDataset) return null;
      let json: any = null;
      try {
        json = await fetchJsonCached(splitMapPath);
      } catch {
        return null;
      }
      // Preferred schema: flat list rows in /id_smiles_y_true_y_pred.json
      if (Array.isArray(json)) {
        const splitById: Record<string, SplitMetaTag> = {};
        const yTrueById: Record<string, number> = {};
        const yPredById: Record<string, number> = {};
        json.forEach((row: any) => {
          if (!row || typeof row !== "object") return;
          const ridRaw = row.real_id ?? row.id;
          if (ridRaw == null) return;
          const rid = String(ridRaw).trim();
          if (!rid) return;
          const splitRaw = String(row.split ?? "").trim().toLowerCase();
          const split = splitRaw === "train" || splitRaw === "test" || splitRaw === "val" ? splitRaw : "";
          if (!split) return;
          const prev = splitById[rid];
          splitById[rid] = prev && prev !== split ? "both" : (split as SplitTag);
          const yt = Number(row.y_true);
          const yp = Number(row.y_pred);
          if (Number.isFinite(yt)) yTrueById[rid] = yt;
          if (Number.isFinite(yp)) yPredById[rid] = yp;
        });
        return { splitById, yTrueById, yPredById };
      }
      // Backward-compatible schema from backend /real-id-split-map
      const map = json?.datasets?.[splitMapDataset]?.split_by_real_id;
      if (!map || typeof map !== "object") return null;
      return { splitById: map as Record<string, SplitMetaTag> };
    };

    const applyMergedRows = (
      merged: ScatterPoint[],
      splitById: Record<string, SplitMetaTag>,
      yTrueById: Record<string, number>,
      yPredById: Record<string, number>
    ) => {
      const visibleRows = merged.filter((d) => d.split !== "val");
      splitByIdRef.current = splitById;
      const splitKinds = new Set(visibleRows.map((d) => d.split).filter((x): x is SplitTag => !!x));
      const canUseSplit = splitKinds.size > 0;
      setVisibleSplits(Array.from(splitKinds).sort());
      setHasSplitData(canUseSplit);
      const yTrueByIdNext: Record<string, number> = {};
      const yPredByIdNext: Record<string, number> = {};
      const yAbsErrByIdNext: Record<string, number> = {};
      const yTrueVals: number[] = [];
      const yPredVals: number[] = [];
      const yAbsErrVals: number[] = [];
      visibleRows.forEach((d) => {
        const rid = String(d.real_id);
        const keys = idLookupKeys(rid);
        const yt = Number(
          keys.map((k) => yTrueById?.[k]).find((v) => Number.isFinite(Number(v))) ?? d.y_true
        );
        const yp = Number(
          keys.map((k) => yPredById?.[k]).find((v) => Number.isFinite(Number(v))) ?? d.y_pred
        );
        if (Number.isFinite(yt)) {
          yTrueByIdNext[rid] = yt;
          yTrueVals.push(yt);
        }
        if (Number.isFinite(yp)) {
          yPredByIdNext[rid] = yp;
          yPredVals.push(yp);
        }
        if (Number.isFinite(yt) && Number.isFinite(yp)) {
          const logAbsErr = Math.log(Math.abs(yt - yp) + 1e-6);
          yAbsErrByIdNext[rid] = logAbsErr;
          yAbsErrVals.push(logAbsErr);
        }
      });
      const yAllVals = [...yTrueVals, ...yPredVals];
      yTrueByIdRef.current = yTrueByIdNext;
      yPredByIdRef.current = yPredByIdNext;
      yAbsErrByIdRef.current = yAbsErrByIdNext;
      setYTrueRange(yTrueVals.length ? { min: Math.min(...yTrueVals), max: Math.max(...yTrueVals) } : null);
      setYPredRange(yPredVals.length ? { min: Math.min(...yPredVals), max: Math.max(...yPredVals) } : null);
      setYAbsErrRange(yAbsErrVals.length ? { min: Math.min(...yAbsErrVals), max: Math.max(...yAbsErrVals) } : null);
      setYSharedRange(yAllVals.length ? { min: Math.min(...yAllVals), max: Math.max(...yAllVals) } : null);
      if (!canUseSplit && colorMode === "split") setColorMode("cluster");
      if (colorMode === "y_true" && yTrueVals.length === 0) setColorMode("cluster");
      if (colorMode === "y_pred" && yPredVals.length === 0) setColorMode("cluster");
      if (colorMode === "y_abs_err" && yAbsErrVals.length === 0) setColorMode("cluster");

      lastRows = visibleRows;
      rowsRef.current = visibleRows;
      const clusterIds = Array.from(new Set(visibleRows.map((d) => d.cluster_id))).sort((a, b) => a - b);
      const idsByCluster: Record<number, string[]> = {};
      visibleRows.forEach((d) => {
        const cid = d.cluster_id;
        if (!idsByCluster[cid]) idsByCluster[cid] = [];
        idsByCluster[cid].push(String(d.real_id));
      });
      Object.keys(idsByCluster).forEach((k) => {
        const cid = Number(k);
        idsByCluster[cid] = Array.from(new Set(idsByCluster[cid]));
      });
      setLegendClusters(clusterIds);
      setLegendClusterIds(idsByCluster);
      draw();
      requestAnimationFrame(() => repaintBaseFillForMode(colorModeRef.current));
    };

    if (compactOverview && Array.isArray(compactOverview?.cells)) {
      try {
        const cells = compactOverview.cells as any[];
        const modelNorm = String(modelKey ?? "").trim();
        const layerNorm = normalizeLayerKey(layerKey);
        const selectedCell =
          cells.find(
            (c: any) =>
              String(c?.epoch ?? "").trim() === modelNorm &&
              normalizeLayerKey(c?.layer) === layerNorm
          ) ??
          cells.find((c: any) => String(c?.epoch ?? "").trim() === modelNorm) ??
          cells.find((c: any) => normalizeLayerKey(c?.layer) === layerNorm) ??
          cells[0];
        const drObj = selectedCell?.dr ?? {};
        const method = projection ?? "pca";
        const coordsMap =
          drObj?.[method] ??
          drObj?.default ??
          drObj?.pca ??
          drObj?.umap ??
          drObj?.tsne ??
          drObj?.mds ??
          {};
        const clusterLabels = selectedCell?.cluster_labels ?? {};
        const moleculeLabels: any[] = Array.isArray(compactOverview?.molecule_labels) ? compactOverview.molecule_labels : [];
        const clusterByAlias = new Map<string, number>();
        const addAlias = (keyRaw: unknown, cid: number) => {
          const key = String(keyRaw ?? "").trim();
          if (!key) return;
          idLookupKeys(key).forEach((k) => {
            if (!clusterByAlias.has(k)) clusterByAlias.set(k, cid);
          });
          const parsed = key.includes("::") ? key.slice(key.indexOf("::") + 2) : key;
          idLookupKeys(parsed).forEach((k) => {
            if (!clusterByAlias.has(k)) clusterByAlias.set(k, cid);
          });
          const trainTagged = `train::${parsed}`;
          const testTagged = `test::${parsed}`;
          const valTagged = `val::${parsed}`;
          idLookupKeys(trainTagged).forEach((k) => {
            if (!clusterByAlias.has(k)) clusterByAlias.set(k, cid);
          });
          idLookupKeys(testTagged).forEach((k) => {
            if (!clusterByAlias.has(k)) clusterByAlias.set(k, cid);
          });
          idLookupKeys(valTagged).forEach((k) => {
            if (!clusterByAlias.has(k)) clusterByAlias.set(k, cid);
          });
        };
        Object.entries(clusterLabels as Record<string, unknown>).forEach(([molId, cidRaw]) => {
          const cid = Number(cidRaw);
          if (!Number.isFinite(cid)) return;
          addAlias(molId, cid);
          const molIdx = Number(molId);
          if (Number.isFinite(molIdx) && moleculeLabels[molIdx] != null) {
            addAlias(String(moleculeLabels[molIdx]), cid);
          }
        });

        const resolveClusterForPoint = (ridRaw: string): number => {
          const rid = String(ridRaw ?? "").trim();
          if (!rid) return 0;
          const aliases = new Set<string>();
          idLookupKeys(rid).forEach((k) => aliases.add(k));
          const stripped = rid.includes("::") ? rid.slice(rid.indexOf("::") + 2) : rid;
          idLookupKeys(stripped).forEach((k) => aliases.add(k));
          for (const key of aliases) {
            const cid = clusterByAlias.get(key);
            if (Number.isFinite(cid)) return Number(cid);
          }
          return 0;
        };

        const splitById: Record<string, SplitMetaTag> = {};
        const splitNames = Array.isArray(compactOverview?.split_names) ? compactOverview.split_names : [];
        const splits = Array.isArray(compactOverview?.splits) ? compactOverview.splits : [];
        splitNames.forEach((name: any, idx: number) => {
          const ids = Array.isArray(splits[idx]) ? splits[idx] : [];
          const normalized: SplitTag | null =
            String(name) === "train" ? "train" : String(name) === "test" ? "test" : String(name) === "val" ? "val" : null;
          if (!normalized) return;
          ids.forEach((id: any) => {
            const key = String(id);
            const prev = splitById[key];
            splitById[key] = prev && prev !== normalized ? "both" : normalized;
          });
        });

        const yTrueById: Record<string, number> = yTrueLookup ?? {};
        const yPredById: Record<string, number> = yPredLookup ?? {};
        const resolveMetricForPoint = (
          lookup: Record<string, number> | undefined,
          ridRaw: string,
          split: SplitTag | undefined
        ): number | undefined => {
          const rid = String(ridRaw ?? "").trim();
          if (!rid || !lookup) return undefined;
          const candidateKeys = new Set<string>();
          idLookupKeys(rid).forEach((k) => candidateKeys.add(k));

          const ridNum = Number(rid);
          const ridLabel =
            Number.isFinite(ridNum) && moleculeLabels[ridNum] != null
              ? String(moleculeLabels[ridNum]).trim()
              : "";
          if (ridLabel) {
            idLookupKeys(ridLabel).forEach((k) => candidateKeys.add(k));
          }

          if (split === "train" || split === "test" || split === "val") {
            idLookupKeys(`${split}::${rid}`).forEach((k) => candidateKeys.add(k));
            if (ridLabel) {
              idLookupKeys(`${split}::${ridLabel}`).forEach((k) => candidateKeys.add(k));
            }
            if (split === "val") {
              idLookupKeys(`test::${rid}`).forEach((k) => candidateKeys.add(k));
              if (ridLabel) {
                idLookupKeys(`test::${ridLabel}`).forEach((k) => candidateKeys.add(k));
              }
            }
          }

          for (const key of candidateKeys) {
            const v = Number(lookup[key]);
            if (Number.isFinite(v)) return v;
          }
          return undefined;
        };
        const merged = Object.entries(coordsMap)
          .map(([rid, pair]: [string, any]) => {
            const xy = Array.isArray(pair) ? pair : [];
            const x = Number(xy[0]);
            const y = Number(xy[1]);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            const ridRaw = String(rid).trim();
            const ridNum = Number(ridRaw);
            const ridLabel =
              Number.isFinite(ridNum) && moleculeLabels[ridNum] != null
                ? String(moleculeLabels[ridNum]).trim()
                : ridRaw;
            const splitRaw = splitById[ridRaw] ?? splitById[ridLabel];
            const split = splitRaw === "train" || splitRaw === "test" || splitRaw === "val" ? splitRaw : undefined;
            if (dataSplit === "train" && split !== "train") return null;
            if (dataSplit === "test" && split !== "test") return null;
            const yTrue = resolveMetricForPoint(yTrueById, ridLabel || ridRaw, split);
            const yPred = resolveMetricForPoint(yPredById, ridLabel || ridRaw, split);
            const splitTag = split === "val" ? "test" : split;
            const displayRid =
              dataSplit === "all" && splitTag ? `${splitTag}::${ridLabel || ridRaw}` : (ridLabel || ridRaw);
            return {
              real_id: displayRid,
              x,
              y,
              cluster_id: resolveClusterForPoint(ridLabel || ridRaw),
              split,
              y_true: Number.isFinite(Number(yTrue)) ? Number(yTrue) : undefined,
              y_pred: Number.isFinite(Number(yPred)) ? Number(yPred) : undefined,
            } as ScatterPoint;
          })
          .filter((d: ScatterPoint | null): d is ScatterPoint => !!d);
        applyMergedRows(merged, splitById, yTrueById, yPredById);
      } catch (err) {
        console.error(err);
        if (container) container.innerHTML = "<div style='color:red'>Failed to load scatter data.</div>";
      }
    } else {
      if (!dataPathXY || !dataPathCluster) {
        if (container) container.innerHTML = "<div style='color:red'>Missing scatter data source.</div>";
      } else {
        const requests: Array<Promise<any>> = [
          fetchScatterJson(dataPathXY, "xy"),
          fetchScatterJson(dataPathCluster, "cluster"),
          fetchSplitMap(),
        ];
        if (secondaryDataPathXY && secondaryDataPathCluster) {
          requests.push(fetchScatterJson(secondaryDataPathXY, "xy"));
          requests.push(fetchScatterJson(secondaryDataPathCluster, "cluster"));
        }

        Promise.all(requests)
          .then((allJson) => {
            const [xyJson, clusterJson, splitMeta, secondaryXYJson, secondaryClusterJson] = allJson;
            const splitById = splitMeta?.splitById ?? {};
            const yTrueById = splitMeta?.yTrueById ?? {};
            const yPredById = splitMeta?.yPredById ?? {};

            const parseRows = (
              xySource: any,
              clusterSource: any,
              split: SplitTag | undefined
            ): ScatterPoint[] => {
              const xyRows = Array.isArray(xySource) ? xySource : [];
              const clusterRows = Array.isArray(clusterSource) ? clusterSource : [];
              const clusterById = new Map<string, number>();
              clusterRows.forEach((d: any) => {
                const id = String(d?.real_id ?? "");
                if (!id) return;
                clusterById.set(id, Number(d?.cluster_id ?? 0) || 0);
              });
              return xyRows
                .map((d: any) => {
                  const id = String(d?.real_id ?? "");
                  if (!id) return null;
                  const clusterFromMap = clusterById.get(id);
                  const clusterFromRow = Number(d?.cluster_id ?? 0) || 0;
                  const splitFromMapRaw = (() => {
                    const keys = idLookupKeys(id);
                    for (let i = 0; i < keys.length; i += 1) {
                      const v = splitById?.[keys[i]];
                      if (v) return v;
                    }
                    return undefined;
                  })();
                  const splitFromMap =
                    splitFromMapRaw === "train" || splitFromMapRaw === "test" || splitFromMapRaw === "val"
                      ? splitFromMapRaw
                      : undefined;
                  return {
                    real_id: id,
                    x: Number(d?.x ?? 0) || 0,
                    y: Number(d?.y ?? 0) || 0,
                    cluster_id: clusterFromMap ?? clusterFromRow,
                    split: splitFromMap ?? split,
                  } as ScatterPoint;
                })
                .filter((d: ScatterPoint | null): d is ScatterPoint => !!d);
            };

            const primaryDefaultSplit: SplitTag | undefined =
              secondaryXYJson ? "train" : (dataSplit === "train" || dataSplit === "test" ? dataSplit : undefined);
            const primaryRows = parseRows(xyJson, clusterJson, primaryDefaultSplit);
            const secondaryRows = secondaryXYJson
              ? parseRows(secondaryXYJson, secondaryClusterJson, "test")
              : [];
            const merged: ScatterPoint[] = [...primaryRows, ...secondaryRows];
            applyMergedRows(merged, splitById, yTrueById, yPredById);
          })
          .catch((err) => {
            console.error(err);
            if (container) container.innerHTML = "<div style='color:red'>Failed to load scatter data.</div>";
          });
      }
    }

    return () => {
      tooltip.remove();
      ro.disconnect();
    };
  }, [
    compactOverview,
    dataPathXY,
    dataPathCluster,
    secondaryDataPathXY,
    secondaryDataPathCluster,
    splitMapPath,
    splitMapDataset,
    dataSplit,
    yTrueLookup,
    yPredLookup,
    modelKey,
    layerKey,
    projection,
  ]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;
    applySelectionStateToSvg(svg);
  }, [activeCluster, activeIds, highlightId, colorMode, labelById, labelBySmiles, labelColors, ySharedRange, yAbsErrRange, getSmilesById]);

  return (
    <div className="w-full h-full flex flex-col">
      <div className="shrink-0 border-b border-zinc-200 bg-white/90 px-1.5 py-1">
        <div className="flex items-center justify-between gap-2">
          <div className="text-sm font-semibold text-zinc-700 leading-4">Scatterplot</div>
          {labelFileName && (
            <span className="text-[11px] text-zinc-500 truncate max-w-[190px]" title={labelFileName}>
              {labelFileName}
            </span>
          )}
        </div>
        <div className="mt-1 flex items-center gap-1.5 flex-nowrap overflow-x-auto whitespace-nowrap">
          {projection && (
            <>
              <select
                value={projection}
                onChange={(e) => onProjectionChange?.(e.target.value as "pca" | "mds" | "umap" | "tsne")}
                className="text-[11px] leading-4 border border-zinc-300 rounded px-1.5 py-0.5 bg-white text-zinc-800"
              >
                <option value="pca">PCA</option>
                <option value="mds">MDS</option>
                <option value="umap">UMAP</option>
                <option value="tsne">t-SNE</option>
              </select>
            </>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleLabelFileUpload(file);
              e.currentTarget.value = "";
            }}
          />
          <button
            type="button"
            className={`px-1.5 py-0.5 text-[11px] leading-4 rounded border ${
              colorMode === "cluster" ? "border-blue-600 text-blue-700 bg-blue-50" : "border-zinc-300 text-zinc-700 bg-white"
            }`}
            onClick={() => setColorModeAndRefresh("cluster")}
          >
            cluster
          </button>
          <button
            type="button"
            className={`px-1.5 py-0.5 text-[11px] leading-4 rounded border ${
              colorMode === "label" ? "border-blue-600 text-blue-700 bg-blue-50" : "border-zinc-300 text-zinc-700 bg-white"
            } disabled:opacity-50`}
            onClick={() => setColorModeAndRefresh("label")}
            disabled={Object.keys(labelById).length === 0 && Object.keys(labelBySmiles).length === 0}
            title={
              Object.keys(labelById).length === 0 && Object.keys(labelBySmiles).length === 0
                ? "Upload a CSV with label column first"
                : "Use uploaded label colors"
            }
          >
            labelled
          </button>
          <button
            type="button"
            className={`px-1.5 py-0.5 text-[11px] leading-4 rounded border ${
              colorMode === "split" ? "border-blue-600 text-blue-700 bg-blue-50" : "border-zinc-300 text-zinc-700 bg-white"
            } disabled:opacity-50`}
            onClick={() => setColorModeAndRefresh("split")}
            disabled={!hasSplitData}
            title={hasSplitData ? "Color points by train/test/val split" : "Only available when split data exists"}
          >
            split
          </button>
          <button
            type="button"
            className={`px-1.5 py-0.5 text-[11px] leading-4 rounded border ${
              colorMode === "y_true" ? "border-blue-600 text-blue-700 bg-blue-50" : "border-zinc-300 text-zinc-700 bg-white"
            } disabled:opacity-50`}
            onClick={() => setColorModeAndRefresh("y_true")}
            disabled={!yTrueRange}
            title={yTrueRange ? "Color points by y_true" : "y_true not available"}
          >
            y_true
          </button>
          <button
            type="button"
            className={`px-1.5 py-0.5 text-[11px] leading-4 rounded border ${
              colorMode === "y_pred" ? "border-blue-600 text-blue-700 bg-blue-50" : "border-zinc-300 text-zinc-700 bg-white"
            } disabled:opacity-50`}
            onClick={() => setColorModeAndRefresh("y_pred")}
            disabled={!yPredRange}
            title={yPredRange ? "Color points by y_pred" : "y_pred not available"}
          >
            y_pred
          </button>
          <button
            type="button"
            className={`px-1.5 py-0.5 text-[11px] leading-4 rounded border ${
              colorMode === "y_abs_err" ? "border-blue-600 text-blue-700 bg-blue-50" : "border-zinc-300 text-zinc-700 bg-white"
            } disabled:opacity-50`}
            onClick={() => setColorModeAndRefresh("y_abs_err")}
            disabled={!yAbsErrRange}
            title={yAbsErrRange ? "Color points by ln(abs(y_true - y_pred) + 1e-6)" : "y_true/y_pred not available"}
          >
            log diff
          </button>
          {labelLoadError && <span className="text-[11px] text-red-600 truncate max-w-[240px]">{labelLoadError}</span>}
          {!labelLoadError && labelMatchInfo && (
            <span className="text-[11px] text-emerald-700 truncate max-w-[220px]" title={labelMatchInfo}>
              {labelMatchInfo}
            </span>
          )}
          <div className="inline-flex flex-col items-start leading-none">
            <button
              type="button"
              className="px-0 py-0 text-[11px] leading-4 text-blue-600 underline decoration-1 underline-offset-2 hover:text-blue-700 hover:decoration-2 transition-colors cursor-pointer disabled:opacity-50"
              onClick={downloadDatasetCsv}
              title="Download current scatter dataset as CSV"
            >
              download
            </button>
            <button
              type="button"
              className="mt-0.5 px-0 py-0 text-[11px] leading-4 text-blue-600 underline decoration-1 underline-offset-2 hover:text-blue-700 hover:decoration-2 transition-colors cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              upload
            </button>
          </div>
        </div>
      </div>
      <div ref={containerRef} className="w-full flex-1 min-h-0" />
      <div className="shrink-0 border-t border-zinc-200 bg-white/90 px-2 py-1 overflow-x-auto overflow-y-hidden">
        {colorMode === "cluster" && (
          <div className="flex flex-nowrap gap-x-3 gap-y-1 min-w-max">
            {legendClusters.map((clusterId) => (
              <button
                key={clusterId}
                type="button"
                className="flex items-center gap-1 cursor-pointer bg-transparent border-0 p-0 m-0"
                onClick={() =>
                  onClusterSelect?.({
                    model: modelKey,
                    layer: layerKey,
                    cluster: clusterId,
                    real_ids: legendClusterIds[clusterId] ?? [],
                  })
                }
              >
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length] }}
                />
                <span
                  className={`text-xs ${
                    activeCluster != null && clusterId === activeCluster ? "text-red-600 font-bold" : "text-zinc-700"
                  }`}
                >
                  {clusterId}
                </span>
              </button>
            ))}
          </div>
        )}
        {colorMode === "split" && (
          <div className="flex flex-nowrap gap-x-3 gap-y-1 min-w-max text-[11px] text-zinc-700">
            {visibleSplits.includes("train") && (
              <button
                type="button"
                className="inline-flex items-center gap-1 bg-transparent border-0 p-0 m-0 cursor-pointer"
                onClick={() => emitHighlightIds(idsForSplit("train"))}
                title="Highlight train points"
              >
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: "#f59e0b" }} />
                train
              </button>
            )}
            {visibleSplits.includes("test") && (
              <button
                type="button"
                className="inline-flex items-center gap-1 bg-transparent border-0 p-0 m-0 cursor-pointer"
                onClick={() => emitHighlightIds(idsForSplit("test"))}
                title="Highlight test points"
              >
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: "#2563eb" }} />
                test
              </button>
            )}
            {visibleSplits.includes("val") && (
              <button
                type="button"
                className="inline-flex items-center gap-1 bg-transparent border-0 p-0 m-0 cursor-pointer"
                onClick={() => emitHighlightIds(idsForSplit("val"))}
                title="Highlight val points"
              >
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: "#1e3a8a" }} />
                val
              </button>
            )}
          </div>
        )}
        {colorMode === "label" && (
          <div className="flex flex-nowrap gap-x-3 gap-y-1 min-w-max">
            {labelLegend.map((label) => (
              <button
                key={label}
                type="button"
                className="flex items-center gap-1 bg-transparent border-0 p-0 m-0 cursor-pointer"
                onClick={() => emitHighlightIds(idsForLabel(label))}
                title={`Highlight label: ${label}`}
              >
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: labelColors[label] ?? "#9ca3af" }}
                />
                <span className="text-xs text-zinc-700">{label}</span>
              </button>
            ))}
            {labelLegend.length === 0 && (
              <span className="text-xs text-zinc-500">Upload a CSV with `label` and `real_id`/`id` columns.</span>
            )}
          </div>
        )}
        {colorMode === "y_true" && ySharedRange && (
          <div className="flex items-center gap-2 min-w-max text-[11px] text-zinc-700">
            <span>{ySharedRange.min.toFixed(3)}</span>
            <div
              className="h-2 w-28 rounded"
              style={{ background: "linear-gradient(to right, #ffffd9, #081d58)" }}
            />
            <span>{ySharedRange.max.toFixed(3)}</span>
          </div>
        )}
        {colorMode === "y_pred" && ySharedRange && (
          <div className="flex items-center gap-2 min-w-max text-[11px] text-zinc-700">
            <span>{ySharedRange.min.toFixed(3)}</span>
            <div
              className="h-2 w-28 rounded"
              style={{ background: "linear-gradient(to right, #ffffd9, #081d58)" }}
            />
            <span>{ySharedRange.max.toFixed(3)}</span>
          </div>
        )}
        {colorMode === "y_abs_err" && yAbsErrRange && (
          <div className="flex items-center gap-2 min-w-max text-[11px] text-zinc-700">
            <span>{yAbsErrRange.min.toFixed(3)}</span>
            <div
              className="h-2 w-28 rounded"
              style={{ background: "linear-gradient(to right, #ffffcc, #b10026)" }}
            />
            <span>{yAbsErrRange.max.toFixed(3)}</span>
          </div>
        )}
      </div>
    </div>
  );
}
