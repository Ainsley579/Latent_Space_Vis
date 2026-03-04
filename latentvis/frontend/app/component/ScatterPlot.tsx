"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { CLUSTER_COLORS, HIGHLIGHT_RED } from "../theme/colors";
import { renderSmilesSvg } from "./SmilesStructure";

type ScatterPoint = {
  real_id: string;
  x: number;
  y: number;
  cluster_id: number;
  split?: "train" | "test";
};

type ColorMode = "cluster" | "label" | "split";

type ScatterPlotProps = {
  dataPathXY: string;
  dataPathCluster: string;
  secondaryDataPathXY?: string;
  secondaryDataPathCluster?: string;
  splitMapPath?: string;
  splitMapDataset?: string;
  dataSplit?: "all" | "train" | "test";
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
  onClearSelection?: () => void;
  resolveImageSrc?: (realId: string | number, imageType: string) => string;
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
  dataPathXY,
  dataPathCluster,
  secondaryDataPathXY,
  secondaryDataPathCluster,
  splitMapPath,
  splitMapDataset,
  dataSplit,
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
  onClearSelection,
  resolveImageSrc,
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
  const [legendClusters, setLegendClusters] = useState<number[]>([]);
  const [legendClusterIds, setLegendClusterIds] = useState<Record<number, string[]>>({});
  const [colorMode, setColorMode] = useState<ColorMode>("cluster");
  const [labelById, setLabelById] = useState<Record<string, string>>({});
  const [labelBySmiles, setLabelBySmiles] = useState<Record<string, string>>({});
  const [labelColors, setLabelColors] = useState<Record<string, string>>({});
  const [labelLegend, setLabelLegend] = useState<string[]>([]);
  const [labelFileName, setLabelFileName] = useState<string>("");
  const [labelLoadError, setLabelLoadError] = useState<string | null>(null);
  const [hasSplitData, setHasSplitData] = useState(false);
  const colorModeRef = useRef<ColorMode>("cluster");

  function idLookupKeys(raw: string): string[] {
    const s = String(raw ?? "").trim();
    if (!s) return [];
    const keys = new Set<string>([s]);
    if (/^-?\d+(\.0+)?$/.test(s)) {
      const n = Number(s);
      if (Number.isFinite(n)) keys.add(String(Math.trunc(n)));
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

  function resolveBaseFillByMode(mode: ColorMode, id: string, cluster: number, split?: "train" | "test"): string {
    if (mode === "split") {
      if (split === "test") return "#2563eb";
      if (split === "train") return "#f59e0b";
      return "#9ca3af";
    }
    if (mode === "label") {
      const direct = lookupLabelById(labelById, id);
      if (direct) return labelColors[direct] ?? "#9ca3af";
      const smiles = getSmilesByIdRef.current?.(String(id));
      if (smiles) {
        const bySmiles = labelBySmiles[String(smiles).trim()];
        if (bySmiles) return labelColors[bySmiles] ?? "#9ca3af";
      }
      return "#9ca3af";
    }
    return CLUSTER_COLORS[Math.max(0, cluster) % CLUSTER_COLORS.length];
  }

  function resolveBaseFill(id: string, cluster: number, split?: "train" | "test"): string {
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
      const split = (el.getAttribute("data-split") || "") as "train" | "test" | "";
      const fill = resolveBaseFillByMode(mode, id, cluster, split || undefined);
      el.setAttribute("data-base-fill", fill);
      if (!highlightId || id !== String(highlightId)) {
        el.setAttribute("fill", fill);
      }
    });
  }

  function setColorModeAndRefresh(next: ColorMode) {
    colorModeRef.current = next;
    setColorMode(next);
    requestAnimationFrame(() => repaintBaseFillForMode(next));
  }

  useEffect(() => {
    colorModeRef.current = colorMode;
  }, [colorMode]);

  function handleLabelFileUpload(file: File) {
    setLabelLoadError(null);
    file
      .text()
      .then((text) => {
        const rows = d3.csvParse(text);
        if (!rows.length) throw new Error("CSV is empty.");
        const columns = rows.columns ?? [];
        const labelCol = columns.find((c) => c.trim().toLowerCase() === "label");
        if (!labelCol) throw new Error("CSV must contain a 'label' column.");

        const idCol =
          columns.find((c) => c.trim().toLowerCase() === "real_id") ??
          columns.find((c) => c.trim().toLowerCase() === "id") ??
          columns.find((c) => c.trim().toLowerCase() === "realid") ??
          columns.find((c) => c.trim().toLowerCase() === "mol_id");
        const smilesCol = columns.find((c) => c.trim().toLowerCase() === "smiles");

        const nextLabelById: Record<string, string> = {};
        const nextLabelBySmiles: Record<string, string> = {};
        if (smilesCol) {
          rows.forEach((row) => {
            const smiles = String((row as Record<string, unknown>)[smilesCol] ?? "").trim();
            const label = String((row as Record<string, unknown>)[labelCol] ?? "").trim();
            if (!smiles || !label) return;
            nextLabelBySmiles[smiles] = label;
          });
        }
        if (idCol) {
          rows.forEach((row) => {
            const id = String((row as Record<string, unknown>)[idCol] ?? "").trim();
            const label = String((row as Record<string, unknown>)[labelCol] ?? "").trim();
            if (!id || !label) return;
            idLookupKeys(id).forEach((k) => {
              nextLabelById[k] = label;
            });
          });
        } else {
          const points = rowsRef.current ?? [];
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

        const labels = Array.from(new Set([...Object.values(nextLabelById), ...Object.values(nextLabelBySmiles)]));
        if (!labels.length) throw new Error("No valid id-label rows found.");
        const legend = labels.sort((a, b) => a.localeCompare(b));
        const colors = buildLabelColors(legend);
        const labelToCluster = new Map<string, number>();
        legend.forEach((name, idx) => labelToCluster.set(name, idx));
        const points = rowsRef.current ?? [];
        const userRows: Array<{ real_id: string; user_cluster_id: number }> = [];
        points.forEach((pt) => {
          const rid = String(pt.real_id);
          let label = lookupLabelById(nextLabelById, rid);
          if (!label) {
            const smiles = getSmilesByIdRef.current?.(rid);
            if (smiles) label = nextLabelBySmiles[String(smiles).trim()];
          }
          if (!label) return;
          const cid = labelToCluster.get(label);
          if (cid == null) return;
          userRows.push({ real_id: rid, user_cluster_id: cid });
        });
        setLabelById(nextLabelById);
        setLabelBySmiles(nextLabelBySmiles);
        setLabelColors(colors);
        setLabelLegend(legend);
        setLabelFileName(file.name);
        setColorMode("label");
        onUserClusteringUpload?.({
          fileName: file.name,
          labelNames: legend,
          rows: userRows,
        });
      })
      .catch((err) => {
        setLabelLoadError(err instanceof Error ? err.message : String(err));
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

      const circles = g.append("g")
        .selectAll("circle")
        .data(lastRows)
        .enter()
        .append("circle")
        .attr("data-id", (d) => String(d.real_id))
        .attr("data-cluster", (d) => String(d.cluster_id))
        .attr("data-split", (d) => String(d.split ?? ""))
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
        .style("cursor", "pointer")
        .on("mouseover", (event, d) => {
          const token = ++hoverToken;
          tooltipMeta.html(`<div><b>ID:</b> ${String(d.real_id)}</div>`);
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
        .on("click", (event, d) => {
          event.stopPropagation();
          onPointSelectRef.current?.({
            model: modelKeyRef.current,
            layer: layerKeyRef.current,
            cluster: d.cluster_id,
            real_id: String(d.real_id),
          });
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

    const buildLegacyScatterUrl = (url: string, kind: "xy" | "cluster") => {
      const [path, query = ""] = url.split("?");
      const suffix = kind === "xy" ? "_xy.json" : "_cluster.json";
      if (!path.endsWith(suffix)) return null;
      const legacy = path.slice(0, -suffix.length) + ".json";
      return query ? `${legacy}?${query}` : legacy;
    };

    const fetchScatterJson = async (url: string, kind: "xy" | "cluster") => {
      const legacyUrl = buildLegacyScatterUrl(url, kind);
      if (!url.startsWith("blob:") && legacyUrl && legacyUrl !== url) {
        const legacyRes = await fetch(legacyUrl);
        if (legacyRes.ok) return legacyRes.json();
      }

      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Failed to fetch scatter ${kind} data`);
      }
      return res.json();
    };

    const fetchSplitMap = async () => {
      if (!splitMapPath || !splitMapDataset) return null;
      const res = await fetch(splitMapPath);
      if (!res.ok) return null;
      const json = await res.json();
      const map = json?.datasets?.[splitMapDataset]?.split_by_real_id;
      if (!map || typeof map !== "object") return null;
      return map as Record<string, "train" | "test" | "both">;
    };

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
        const [xyJson, clusterJson, splitById, secondaryXYJson, secondaryClusterJson] = allJson;

          const parseRows = (
          xySource: any,
          clusterSource: any,
          split: "train" | "test" | undefined
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
              const splitFromMap = splitFromMapRaw === "train" || splitFromMapRaw === "test" ? splitFromMapRaw : undefined;
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

        const primaryDefaultSplit: "train" | "test" | undefined =
          secondaryXYJson ? "train" : (dataSplit === "train" || dataSplit === "test" ? dataSplit : undefined);
        const primaryRows = parseRows(xyJson, clusterJson, primaryDefaultSplit);
        const secondaryRows = secondaryXYJson
          ? parseRows(secondaryXYJson, secondaryClusterJson, "test")
          : [];
        const merged: ScatterPoint[] = [...primaryRows, ...secondaryRows];
        const splitKinds = new Set(merged.map((d) => d.split).filter((x): x is "train" | "test" => !!x));
        const canUseSplit = splitKinds.size > 0;
        setHasSplitData(canUseSplit);
        if (!canUseSplit && colorMode === "split") {
          setColorMode("cluster");
        }

        lastRows = merged;
        rowsRef.current = merged;
        const clusterIds = Array.from(new Set(merged.map((d) => d.cluster_id))).sort((a, b) => a - b);
        const idsByCluster: Record<number, string[]> = {};
        merged.forEach((d) => {
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
      })
      .catch((err) => {
        console.error(err);
        if (container) container.innerHTML = "<div style='color:red'>Failed to load scatter data.</div>";
      });

    return () => {
      tooltip.remove();
      ro.disconnect();
    };
  }, [
    dataPathXY,
    dataPathCluster,
    secondaryDataPathXY,
    secondaryDataPathCluster,
    splitMapPath,
    splitMapDataset,
    dataSplit,
  ]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;
    const rows = rowsRef.current;
    if (!rows) return;

    const activeIdSet = activeIds ? new Set(activeIds.map((x) => String(x))) : null;
    const circles = Array.from(svg.querySelectorAll("circle[data-id]")) as SVGCircleElement[];
    circles.forEach((el) => {
      const id = el.getAttribute("data-id") || "";
      const cluster = Number(el.getAttribute("data-cluster") || "0");
      const split = (el.getAttribute("data-split") || "") as "train" | "test" | "";
      const computedBaseFill = resolveBaseFill(id, cluster, split || undefined);
      el.setAttribute("data-base-fill", computedBaseFill);
      const baseFill = computedBaseFill;
      const isHovered = !!highlightId && id === String(highlightId);
      const inActiveIds = !!activeIdSet && activeIdSet.has(id);

      el.setAttribute("r", isHovered ? "5.2" : "3.2");
      el.setAttribute("fill", isHovered ? "#f59e0b" : baseFill);
      if (isHovered) {
        el.setAttribute("fill-opacity", "1");
        el.setAttribute("stroke", "#dc2626");
        el.setAttribute("stroke-width", "1.8");
      } else if (activeIdSet) {
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
      const hitCircle = circles.find((c) => (c.getAttribute("data-id") || "") === targetId);
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
  }, [activeCluster, activeIds, highlightId, colorMode, labelById, labelBySmiles, labelColors]);

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
              <span className="text-[11px] text-zinc-600">Method:</span>
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
            accept=".csv,text/csv"
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
            title={hasSplitData ? "Color points by train/test split" : "Only available in train+test mode"}
          >
            split
          </button>
          {labelLoadError && <span className="text-[11px] text-red-600 truncate max-w-[240px]">{labelLoadError}</span>}
          <button
            type="button"
            className="px-0 py-0 text-[11px] leading-4 text-blue-600 underline decoration-1 underline-offset-2 hover:text-blue-700 hover:decoration-2 transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            upload
          </button>
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
            <span className="inline-flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: "#f59e0b" }} />
              train
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: "#2563eb" }} />
              test
            </span>
          </div>
        )}
        {colorMode === "label" && (
          <div className="flex flex-nowrap gap-x-3 gap-y-1 min-w-max">
            {labelLegend.map((label) => (
              <div key={label} className="flex items-center gap-1">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: labelColors[label] ?? "#9ca3af" }}
                />
                <span className="text-xs text-zinc-700">{label}</span>
              </div>
            ))}
            {labelLegend.length === 0 && (
              <span className="text-xs text-zinc-500">Upload a CSV with `label` and `real_id`/`id` columns.</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
