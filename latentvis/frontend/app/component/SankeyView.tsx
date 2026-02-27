"use client";
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import ZoomPan from "./ZoomPan";
import { CLUSTER_COLORS, SANKEY_2D_COLUMN_LINK_COLOR, SANKEY_2D_ROW_LINK_COLOR } from "../theme/colors";

const DEFAULT_MAX_INSTANCE = 158;
const SHOW_FMI_CSPA = false;

type SankeyEdge = {
  source_node_id?: number;
  target_node_id?: number;
  source_label?: string;
  target_label?: string;
  value?: number;
  pct_of_source?: number;
};

type SankeyNodeLike = {
  id: number;
  label: string;
  value: number;
  previous: number;
};

function parseClusterFromLabel(label: string): number {
  const m = /_C(\d+)$/.exec(label);
  return m ? Number(m[1]) : 0;
}

function parseGroupFromLabel(label: string): string {
  const i = label.lastIndexOf("_C");
  return i === -1 ? label : label.slice(0, i);
}

function buildNodesByGroup(edges: SankeyEdge[]): Record<string, SankeyNodeLike[]> {
  const inSum = new Map<string, number>();
  const outSum = new Map<string, number>();
  const labelsByGroup = new Map<string, Set<string>>();

  edges.forEach((e) => {
    const src = String(e.source_label ?? "");
    const tgt = String(e.target_label ?? "");
    const val = Number(e.value ?? 0) || 0;
    if (!src || !tgt || val <= 0) return;

    outSum.set(src, (outSum.get(src) ?? 0) + val);
    inSum.set(tgt, (inSum.get(tgt) ?? 0) + val);

    const srcGroup = parseGroupFromLabel(src);
    const tgtGroup = parseGroupFromLabel(tgt);
    if (!labelsByGroup.has(srcGroup)) labelsByGroup.set(srcGroup, new Set());
    if (!labelsByGroup.has(tgtGroup)) labelsByGroup.set(tgtGroup, new Set());
    labelsByGroup.get(srcGroup)?.add(src);
    labelsByGroup.get(tgtGroup)?.add(tgt);
  });

  const out: Record<string, SankeyNodeLike[]> = {};
  labelsByGroup.forEach((labels, group) => {
    const sorted = Array.from(labels).sort((a, b) => parseClusterFromLabel(a) - parseClusterFromLabel(b));
    let previous = 0;
    out[group] = sorted.map((label) => {
      const value = Math.max(inSum.get(label) ?? 0, outSum.get(label) ?? 0);
      const node = {
        id: parseClusterFromLabel(label),
        label,
        value,
        previous,
      };
      previous += value;
      return node;
    });
  });

  return out;
}

function normalizeSankeyData(raw: any): any {
  const data = JSON.parse(JSON.stringify(raw ?? {}));

  Object.keys(data.epoch || {}).forEach((rowKey) => {
    const rowObj = data.epoch[rowKey] || {};
    const edges: SankeyEdge[] = Array.isArray(rowObj.edges) ? rowObj.edges : [];
    if (!rowObj.nodes || Object.keys(rowObj.nodes).length === 0) {
      rowObj.nodes = buildNodesByGroup(edges);
    }
    data.epoch[rowKey] = rowObj;
  });

  Object.keys(data.layer || {}).forEach((colKey) => {
    const colObj = data.layer[colKey] || {};
    const edges: SankeyEdge[] = Array.isArray(colObj.edges) ? colObj.edges : [];
    if (!colObj.nodes || Object.keys(colObj.nodes).length === 0) {
      colObj.nodes = buildNodesByGroup(edges);
    }
    data.layer[colKey] = colObj;
  });

  return data;
}

function sortRowKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    const aNum = !Number.isNaN(na) && String(na) === String(a);
    const bNum = !Number.isNaN(nb) && String(nb) === String(b);
    if (aNum && bNum) return na - nb;
    if (aNum) return -1;
    if (bNum) return 1;
    return a.localeCompare(b);
  });
}

const PREFERRED_LAYER_ORDER = [
  "fingerprint",
  "gnn_layer_1",
  "gnn_layer_2",
  "gnn_layer_3",
  "gnn_layer_4",
  "gnn_layer_5",
  "gnn_layer_6",
  "ffn_layer_1",
  "ffn_layer_2",
  "fnn_layer_1",
  "fnn_layer_2",
  "emb",
  "output_block_0",
  "output_block_1",
  "output_block_2",
  "output_block_3",
  "output_block_4",
  "u_mlp_input",
  "u_mlp_l1",
  "u_mlp_l2",
];
const PREFERRED_LAYER_INDEX = new Map(PREFERRED_LAYER_ORDER.map((name, idx) => [name, idx]));

function toShortLayerName(name: string): string {
  if (name === "fingerprint") return "fp";
  const gnn = /^gnn_layer_(\d+)$/.exec(name);
  if (gnn) return `gnn${gnn[1]}`;
  const ffn = /^ffn_layer_(\d+)$/.exec(name);
  if (ffn) return `ffn${ffn[1]}`;
  const fnn = /^fnn_layer_(\d+)$/.exec(name);
  if (fnn) return `fnn${fnn[1]}`;
  return name;
}

function sortLayerKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const ai = PREFERRED_LAYER_INDEX.get(a);
    const bi = PREFERRED_LAYER_INDEX.get(b);
    if (ai != null && bi != null) return ai - bi;
    if (ai != null) return -1;
    if (bi != null) return 1;

    // For unknown layer names, keep numerical suffixes in natural order.
    const am = /^(.*?)(\d+)$/.exec(a);
    const bm = /^(.*?)(\d+)$/.exec(b);
    if (am && bm && am[1] === bm[1]) {
      return Number(am[2]) - Number(bm[2]);
    }
    return a.localeCompare(b);
  });
}

type SankeyViewProps = {
  dataPath?: string;
  metricsPath?: string;
  transitionStatsPath?: string;
  onModelSelect?: (modelKey: string) => void;
  onTransform?: (t: { x: number; y: number; scale: number }) => void;
  externalTransform?: { x: number; y: number; scale: number };
  onLayerSelect?: (layerKey: string) => void;
  onNodeSelect?: (sel: { model: string; layer: string; cluster: number }) => void;
  selectedNode?: { model: string; layer: string; cluster: number } | null;
  activeCell?: { model: string; layer: string } | null;
  onCellSelect?: (active: boolean) => void;
  onSwapAxes?: (swap: boolean) => void;
};

export default function SankeyView({
  dataPath = "/data/dimenet/sankey/all_sankey.json",
  metricsPath = "/data/dimenet/bic_aic/avg_fmi_cspa_by_layer.json",
  transitionStatsPath = "/data/chemprop/clustering_metrics/layer_transition_stats.json",
  onModelSelect,
  onTransform,
  externalTransform,
  onLayerSelect,
  onNodeSelect,
  selectedNode,
  activeCell,
  onCellSelect,
  onSwapAxes,
}: SankeyViewProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const fmiRef = useRef<HTMLDivElement | null>(null);
  const rowMetricRef = useRef<HTMLDivElement | null>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [maxInstanceValue, setMaxInstanceValue] = useState(DEFAULT_MAX_INSTANCE);
  const [fmiData, setFmiData] = useState<Array<{ layer: string; fmi: number; cspa: number }>>([]);
  const [rowMetricData, setRowMetricData] = useState<Array<{ model: string; fmi: number; cspa: number }>>([]);
  const [showBaseSankey, setShowBaseSankey] = useState(true);
  const [showCellBackground, setShowCellBackground] = useState(true);
  const [showGapHeatmap, setShowGapHeatmap] = useState(true);
  const [layout, setLayout] = useState<{
    rowNames: string[];
    colNames: string[];
    cellW: number;
    cellH: number;
    cellStepX: number;
    cellStepY: number;
    innerW: number;
    innerH: number;
    margin: { top: number; left: number; right: number; bottom: number };
    cellMargin: { top: number; right: number; bottom: number; left: number };
  } | null>(null);

  useEffect(() => {
    const container = chartRef.current;
    if (!container) return;
    container.innerHTML = "";

    const urlParams = new URLSearchParams(window.location.search);
    let SWAP_AXES = urlParams.get("swap") === "1";

    function setSwapParam(nextSwap: boolean) {
      onSwapAxes?.(nextSwap);
      const p = new URLSearchParams(window.location.search);
      if (nextSwap) p.set("swap", "1");
      else p.delete("swap");

      const minEl = document.getElementById("minRange") as HTMLInputElement | null;
      const maxEl = document.getElementById("maxRange") as HTMLInputElement | null;
      if (minEl && maxEl) {
        p.set("min", String(minEl.value));
        p.set("max", String(maxEl.value));
      }

      const next = window.location.pathname + (p.toString() ? "?" + p.toString() : "");
      window.location.href = next;
    }

    Promise.all([
      fetch(dataPath).then((res) => {
        if (!res.ok) throw new Error("Failed to fetch all_sankey.json");
        return res.json();
      }),
      fetch(transitionStatsPath)
        .then((res) => (res.ok ? res.json() : null))
        .catch(() => null),
    ])
      .then(([rawData, transitionJson]) => {
        const data = normalizeSankeyData(rawData);
        const ROW_NAMES: string[] = sortRowKeys(Object.keys(data?.epoch || {}));
        const COLUMN_NAMES: string[] = sortLayerKeys(Object.keys(data?.layer || {}));
        const MODEL_SET = new Set(ROW_NAMES);
        const LAYER_SET = new Set(COLUMN_NAMES);

        const ROW_INDEX = new Map(ROW_NAMES.map((d, i) => [d, i]));
        const COLUMN_INDEX = new Map(COLUMN_NAMES.map((d, i) => [d, i]));

        const X_NAMES = SWAP_AXES ? ROW_NAMES : COLUMN_NAMES;
        const Y_NAMES = SWAP_AXES ? COLUMN_NAMES : ROW_NAMES;

        function cellOrigin(rowName: string, columnName: string, stepX: number, stepY: number) {
          const col = SWAP_AXES ? (ROW_INDEX.get(rowName) ?? 0) : (COLUMN_INDEX.get(columnName) ?? 0);
          const row = SWAP_AXES ? (COLUMN_INDEX.get(columnName) ?? 0) : (ROW_INDEX.get(rowName) ?? 0);
          return { col, row, x: col * stepX, y: row * stepY };
        }

        const margin = { top: 20, left: 0, right: 0, bottom: 20 };
        const CELL_W = 120;
        const CELL_H = 150;
        const CELL_GAP = 10;
        const cellStepX = CELL_W + CELL_GAP;
        const cellStepY = CELL_H + CELL_GAP;

        const width =
          margin.left +
          margin.right +
          CELL_W * X_NAMES.length +
          Math.max(0, X_NAMES.length - 1) * CELL_GAP;
        const height =
          margin.top +
          margin.bottom +
          CELL_H * Y_NAMES.length +
          Math.max(0, Y_NAMES.length - 1) * CELL_GAP;

        const svg = d3
          .select(container)
          .append("svg")
          .attr("width", width)
          .attr("height", height)
          .attr("viewBox", `0 0 ${width} ${height}`)
          .attr("preserveAspectRatio", "xMinYMin meet");

        (container.style as any).width = width + "px";
        (container.style as any).height = height + "px";

        const cellW = CELL_W;
        const cellH = CELL_H;
        const cellMargin = { top: 5, right: 5, bottom: 5, left: 5 };
        const innerW = cellW - cellMargin.left - cellMargin.right;
        const innerH = cellH - cellMargin.top - cellMargin.bottom;

        const gRoot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

        setLayout({
          rowNames: Y_NAMES,
          colNames: X_NAMES,
          cellW,
          cellH,
          cellStepX,
          cellStepY,
          innerW,
          innerH,
          margin,
          cellMargin,
        });

        const cellClusterCount = new Map<string, number>();
        ROW_NAMES.forEach((rowName) => {
          COLUMN_NAMES.forEach((columnName) => {
            const nodesForCell = (data.epoch?.[rowName]?.nodes?.[columnName] || []) as any[];
            const ids = new Set<number>();
            nodesForCell.forEach((n: any) => {
              const c =
                n?.cluster != null && n?.cluster !== ""
                  ? Number(n.cluster)
                  : parseClusterFromLabel(String(n?.label ?? ""));
              if (Number.isFinite(c)) ids.add(c);
            });
            cellClusterCount.set(`${rowName}|${columnName}`, ids.size || nodesForCell.length || 0);
          });
        });
        const clusterCounts = Array.from(cellClusterCount.values());
        const minCellCluster = clusterCounts.length ? Math.min(...clusterCounts) : 0;
        const maxCellCluster = clusterCounts.length ? Math.max(...clusterCounts) : 1;
        const cellBgColor = d3
          .scaleLinear<string>()
          .domain(
            minCellCluster === maxCellCluster
              ? [minCellCluster, minCellCluster + 1]
              : [minCellCluster, maxCellCluster]
          )
          .range(["#eff6ff", "#46518C"])
          .clamp(true);

        const cells: d3.Selection<SVGGElement, any, any, any>[] = [];
        let lockedCell: { rowName: string; columnName: string } | null = null;
        X_NAMES.forEach((xKey, col) => {
          Y_NAMES.forEach((yKey, row) => {
            const rowName = SWAP_AXES ? xKey : yKey;
            const columnName = SWAP_AXES ? yKey : xKey;

            const g = gRoot
              .append("g")
              .attr("id", `cell-${rowName}-${columnName}`)
              .attr("transform", `translate(${col * cellStepX}, ${row * cellStepY})`)
              .datum({ rowName, columnName, col, row });

            const cellCount = cellClusterCount.get(`${rowName}|${columnName}`) ?? 0;
            g.append("rect")
              .attr("width", cellW)
              .attr("height", cellH)
              .attr("fill", showCellBackground ? cellBgColor(cellCount) : "white")
              .attr("stroke", "#888")
              .attr("data-cluster-count", cellCount);

            const innerG = g.append("g").attr("transform", `translate(${cellMargin.left}, ${cellMargin.top})`);
            cells.push(innerG);

            g.on("click", (_event: any, datum: any) => {
                const isSameLocked =
                  lockedCell?.rowName === datum.rowName &&
                  lockedCell?.columnName === datum.columnName;

                // Toggle off when clicking the same locked cell.
                if (isSameLocked) {
                  lockedCell = null;
                  setCellHighlight(datum.rowName, datum.columnName, false);
                  onCellSelect?.(false);
                  return;
                }

                lockedCell = { rowName: datum.rowName, columnName: datum.columnName };
                setCellHighlight(datum.rowName, datum.columnName, true);

                const modelCandidate = MODEL_SET.has(datum.rowName)
                  ? datum.rowName
                  : MODEL_SET.has(datum.columnName)
                    ? datum.columnName
                    : null;
                if (modelCandidate) {
                  onModelSelect?.(modelCandidate);
                }
                const layerCandidate = LAYER_SET.has(datum.rowName)
                  ? datum.rowName
                  : LAYER_SET.has(datum.columnName)
                    ? datum.columnName
                    : null;
                if (layerCandidate) {
                  onLayerSelect?.(layerCandidate);
                }
                // Fire after model/layer updates so linked Sankey highlights follow the new cell immediately.
                onCellSelect?.(true);
              });
          });
        });


        const maxClusters = (() => {
          let max = 0;
          const epochEntries = data.epoch || {};
          Object.keys(epochEntries).forEach((rowKey) => {
            const nodesByColumn = epochEntries[rowKey]?.nodes || {};
            Object.keys(nodesByColumn).forEach((columnKey) => {
              const list = nodesByColumn[columnKey] || [];
              if (list.length > max) max = list.length;
            });
          });
          return Math.max(1, max);
        })();

        const maxInstance = (() => {
          let max = 0;
          const epochEntries = data.epoch || {};
          Object.keys(epochEntries).forEach((rowKey) => {
            const nodesByColumn = epochEntries[rowKey]?.nodes || {};
            Object.keys(nodesByColumn).forEach((columnKey) => {
              const list = nodesByColumn[columnKey] || [];
              list.forEach((node: any) => {
                const val = Number(node?.value ?? 0);
                const prev = Number(node?.previous ?? 0);
                const total = prev + val;
                if (total > max) max = total;
              });
            });
          });
          return Math.max(1, max);
        })();
        setMaxInstanceValue(maxInstance);

        const x = d3.scaleBand<number>().range([0, innerW]).domain(d3.range(maxClusters)).padding(0.2);
        const y = d3.scaleLinear().domain([0, maxInstance]).range([innerH, 0]);
        const MAX_NODE_BAR_WIDTH = 20;
        const nodeBarWidth = Math.min(x.bandwidth(), MAX_NODE_BAR_WIDTH);
        const nodeBarOffset = (x.bandwidth() - nodeBarWidth) / 2;

        const transitionEntriesByEpoch: Array<{
          epoch: string;
          from_layer: string;
          to_layer: string;
          pct_changed: number;
        }> = Array.isArray((transitionJson as any)?.by_epoch) ? (transitionJson as any).by_epoch : [];
        const transitionEntriesByLayer: Array<{
          layer: string;
          from_epoch: string;
          to_epoch: string;
          pct_changed: number;
        }> = Array.isArray((transitionJson as any)?.by_layer) ? (transitionJson as any).by_layer : [];
        const pctByEpochTransition = new Map<string, number>();
        transitionEntriesByEpoch.forEach((d) => {
          const epoch = String(d?.epoch ?? "").trim();
          const fromLayer = String(d?.from_layer ?? "").trim();
          const toLayer = String(d?.to_layer ?? "").trim();
          const pct = Number(d?.pct_changed ?? 0);
          if (!epoch || !fromLayer || !toLayer || !Number.isFinite(pct)) return;
          pctByEpochTransition.set(`${epoch}|${fromLayer}|${toLayer}`, Math.max(0, pct));
        });
        const pctByLayerTransition = new Map<string, number>();
        transitionEntriesByLayer.forEach((d) => {
          const layer = String(d?.layer ?? "").trim();
          const fromEpoch = String(d?.from_epoch ?? "").trim();
          const toEpoch = String(d?.to_epoch ?? "").trim();
          const pct = Number(d?.pct_changed ?? 0);
          if (!layer || !fromEpoch || !toEpoch || !Number.isFinite(pct)) return;
          pctByLayerTransition.set(`${layer}|${fromEpoch}|${toEpoch}`, Math.max(0, pct));
        });
        const maxGapPct = Math.max(
          1,
          ...Array.from(pctByEpochTransition.values()),
          ...Array.from(pctByLayerTransition.values())
        );
        const redGapColor = d3
          .scaleLinear<string>()
          .domain([0, maxGapPct])
          .range(["#f3f4f6", "#7f1d1d"])
          .clamp(true);
        const greenGapColor = d3
          .scaleLinear<string>()
          .domain([0, maxGapPct])
          .range(["#f0fdf4", "#166534"])
          .clamp(true);
        const horizontalGapColor = SWAP_AXES ? greenGapColor : redGapColor;
        const verticalGapColor = SWAP_AXES ? redGapColor : greenGapColor;
        if (showGapHeatmap) {
          const gapHeatmap = gRoot.append("g").attr("class", "cell-gap-heatmap");
          if (!SWAP_AXES) {
          // Horizontal gaps: layer-to-layer transitions for each epoch (by_epoch)
          ROW_NAMES.forEach((epoch, rowIdx) => {
            for (let i = 0; i < COLUMN_NAMES.length - 1; i += 1) {
              const fromLayer = COLUMN_NAMES[i];
              const toLayer = COLUMN_NAMES[i + 1];
              const pct = pctByEpochTransition.get(`${epoch}|${fromLayer}|${toLayer}`) ?? 0;
              const xGap = i * cellStepX + cellW;
              const yGap = rowIdx * cellStepY;
              gapHeatmap
                .append("rect")
                .attr("x", xGap)
                .attr("y", yGap)
                .attr("width", CELL_GAP)
                .attr("height", cellH)
                .attr("fill", horizontalGapColor(pct))
                .attr("stroke", "#ffffff")
                .attr("stroke-width", 0.7)
                .append("title")
                .text(`${epoch}: ${fromLayer} -> ${toLayer}, ${pct.toFixed(2)}%`);
            }
          });
          // Vertical gaps: epoch-to-epoch transitions for each layer (by_layer)
          COLUMN_NAMES.forEach((layer, colIdx) => {
            for (let i = 0; i < ROW_NAMES.length - 1; i += 1) {
              const fromEpoch = ROW_NAMES[i];
              const toEpoch = ROW_NAMES[i + 1];
              const pct = pctByLayerTransition.get(`${layer}|${fromEpoch}|${toEpoch}`) ?? 0;
              const xGap = colIdx * cellStepX;
              const yGap = i * cellStepY + cellH;
              gapHeatmap
                .append("rect")
                .attr("x", xGap)
                .attr("y", yGap)
                .attr("width", cellW)
                .attr("height", CELL_GAP)
                .attr("fill", verticalGapColor(pct))
                .attr("stroke", "#ffffff")
                .attr("stroke-width", 0.7)
                .append("title")
                .text(`${layer}: ${fromEpoch} -> ${toEpoch}, ${pct.toFixed(2)}%`);
            }
          });
          } else {
          // Horizontal gaps: epoch-to-epoch transitions for each layer (by_layer)
          COLUMN_NAMES.forEach((layer, rowIdx) => {
            for (let i = 0; i < ROW_NAMES.length - 1; i += 1) {
              const fromEpoch = ROW_NAMES[i];
              const toEpoch = ROW_NAMES[i + 1];
              const pct = pctByLayerTransition.get(`${layer}|${fromEpoch}|${toEpoch}`) ?? 0;
              const xGap = i * cellStepX + cellW;
              const yGap = rowIdx * cellStepY;
              gapHeatmap
                .append("rect")
                .attr("x", xGap)
                .attr("y", yGap)
                .attr("width", CELL_GAP)
                .attr("height", cellH)
                .attr("fill", horizontalGapColor(pct))
                .attr("stroke", "#ffffff")
                .attr("stroke-width", 0.7)
                .append("title")
                .text(`${layer}: ${fromEpoch} -> ${toEpoch}, ${pct.toFixed(2)}%`);
            }
          });
          // Vertical gaps: layer-to-layer transitions for each epoch (by_epoch)
          ROW_NAMES.forEach((epoch, colIdx) => {
            for (let i = 0; i < COLUMN_NAMES.length - 1; i += 1) {
              const fromLayer = COLUMN_NAMES[i];
              const toLayer = COLUMN_NAMES[i + 1];
              const pct = pctByEpochTransition.get(`${epoch}|${fromLayer}|${toLayer}`) ?? 0;
              const xGap = colIdx * cellStepX;
              const yGap = i * cellStepY + cellH;
              gapHeatmap
                .append("rect")
                .attr("x", xGap)
                .attr("y", yGap)
                .attr("width", cellW)
                .attr("height", CELL_GAP)
                .attr("fill", verticalGapColor(pct))
                .attr("stroke", "#ffffff")
                .attr("stroke-width", 0.7)
                .append("title")
                .text(`${epoch}: ${fromLayer} -> ${toLayer}, ${pct.toFixed(2)}%`);
            }
          });
          }
        }

        const nodeLookup = new Map<string, { rowName: string; columnName: string; idx: number; node: any }>();
        const nodeIdLookup = new Map<string, { rowName: string; columnName: string; idx: number; node: any }>();

        function columnFromLabel(label: string) {
          const i = label.lastIndexOf("_C");
          if (i === -1) return label;
          return label.slice(0, i);
        }

        function canonicalLayerName(name: string): string {
          if (/^fnn_layer_/.test(name)) return name.replace(/^fnn_layer_/, "ffn_layer_");
          return name;
        }

        function layerNameAliases(name: string): string[] {
          const out = new Set<string>([name, canonicalLayerName(name)]);
          if (/^ffn_layer_/.test(name)) out.add(name.replace(/^ffn_layer_/, "fnn_layer_"));
          if (/^fnn_layer_/.test(name)) out.add(name.replace(/^fnn_layer_/, "ffn_layer_"));
          return Array.from(out);
        }

        function lookupNode(
          rowName: string,
          columnName: string,
          label: string,
          nodeId: number | undefined
        ): { rowName: string; columnName: string; idx: number; node: any } | null {
          for (const colName of layerNameAliases(columnName)) {
            const byLabel = nodeLookup.get(`${rowName}|${colName}|${label}`);
            if (byLabel) return byLabel;
            if (nodeId !== undefined) {
              const byId = nodeIdLookup.get(`${rowName}|${colName}|${String(nodeId)}`);
              if (byId) return byId;
            }
          }
          return null;
        }

        const nodeDrawQueue: Array<{
          g: d3.Selection<SVGGElement, any, any, any>;
          rowName: string;
          columnName: string;
          nodesForCell: any[];
        }> = [];
        cells.forEach((g) => {
          const { rowName, columnName } = g.datum();
          const rowData = data.epoch?.[rowName];
          const nodesForCell = (rowData && rowData.nodes && rowData.nodes[columnName]) || [];
          nodeDrawQueue.push({ g, rowName, columnName, nodesForCell });

          nodesForCell.forEach((node: any, idx: number) => {
            const label = String(node.label ?? "");
            const key = `${rowName}|${columnName}|${label}`;
            nodeLookup.set(key, { rowName, columnName, idx, node });
            const nodeId = node?.id;
            if (nodeId !== undefined && nodeId !== null) {
              const idKey = `${rowName}|${columnName}|${String(nodeId)}`;
              nodeIdLookup.set(idKey, { rowName, columnName, idx, node });
            }
          });
        });

        const allEdgeValues: number[] = [];
        Object.keys(data.epoch || {}).forEach((k) => {
          (data.epoch[k].edges || []).forEach((e: any) => allEdgeValues.push(+e.value || 0));
        });
        Object.keys(data.layer || {}).forEach((k) => {
          (data.layer[k].edges || []).forEach((e: any) => allEdgeValues.push(+e.value || 0));
        });
        const maxEdgeVal = d3.max(allEdgeValues) || 1;
        const linkWidthScale = d3.scaleSqrt().domain([0, maxEdgeVal]).range([0.5, 20]).clamp(true);

        function appendLinkPath(
          group: d3.Selection<SVGGElement, unknown, null, undefined>,
          pathD: string,
          edgeValue: number,
          strokeW: number,
          strokeColor: string,
          opacity: number
        ) {
          return group
            .append("path")
            .attr("d", pathD)
            .attr("fill", "none")
            .attr("stroke", strokeColor)
            .attr("stroke-width", strokeW)
            .attr("stroke-opacity", opacity)
            .attr("class", "link-path")
            .attr("data-value", +edgeValue || 0);
        }

        function buildGreedyBridgeLinks(
          sourceNodes: any[],
          targetNodes: any[]
        ): Array<{ sourceIdx: number; targetIdx: number; value: number }> {
          const src = sourceNodes
            .map((n, i) => ({ i, cluster: Number(n?.id ?? i), rem: Math.max(0, Number(n?.value ?? 0)) }))
            .filter((d) => d.rem > 0)
            .sort((a, b) => a.cluster - b.cluster);
          const tgt = targetNodes
            .map((n, i) => ({ i, cluster: Number(n?.id ?? i), rem: Math.max(0, Number(n?.value ?? 0)) }))
            .filter((d) => d.rem > 0)
            .sort((a, b) => a.cluster - b.cluster);

          const out: Array<{ sourceIdx: number; targetIdx: number; value: number }> = [];
          let si = 0;
          let ti = 0;
          while (si < src.length && ti < tgt.length) {
            const amount = Math.min(src[si].rem, tgt[ti].rem);
            if (amount > 0) {
              out.push({ sourceIdx: src[si].i, targetIdx: tgt[ti].i, value: amount });
              src[si].rem -= amount;
              tgt[ti].rem -= amount;
            }
            if (src[si].rem <= 1e-9) si += 1;
            if (tgt[ti].rem <= 1e-9) ti += 1;
          }
          return out;
        }

        let linkGroup: d3.Selection<SVGGElement, unknown, null, undefined> | null = null;
        let linkGroup2: d3.Selection<SVGGElement, unknown, null, undefined> | null = null;
        if (showBaseSankey) {
          // column-links (per layer) -> vertical inside a column
          linkGroup = gRoot.append("g").attr("class", "links column-links");
          Object.keys(data.layer || {}).forEach((columnKey) => {
          const columnObj = data.layer[columnKey];
          if (COLUMN_INDEX.get(columnKey) == null) return;

          (columnObj.edges || []).forEach((edge: any) => {
            let sourceInfo: any = null;
            let targetInfo: any = null;
            const nodesByRow = columnObj.nodes || {};
            const sourceLabel = String(edge.source_label ?? "");
            const targetLabel = String(edge.target_label ?? "");

            Object.keys(nodesByRow).forEach((rowName) => {
              (nodesByRow[rowName] || []).forEach((node: any, idx: number) => {
                const nodeLabel = String(node?.label ?? "");
                const sourceByLabel = sourceLabel !== "" && nodeLabel === sourceLabel;
                const targetByLabel = targetLabel !== "" && nodeLabel === targetLabel;
                const sourceById =
                  sourceLabel === "" && edge.source_node_id !== undefined && node?.id === edge.source_node_id;
                const targetById =
                  targetLabel === "" && edge.target_node_id !== undefined && node?.id === edge.target_node_id;

                if (sourceByLabel || sourceById) sourceInfo = { node, rowName, idx };
                if (targetByLabel || targetById) targetInfo = { node, rowName, idx };
              });
            });

            if (!sourceInfo || !targetInfo) return;

            const srcRowIdxRaw = ROW_INDEX.get(sourceInfo.rowName);
            const tgtRowIdxRaw = ROW_INDEX.get(targetInfo.rowName);
            if (srcRowIdxRaw == null || tgtRowIdxRaw == null) return;
            if (Math.abs(tgtRowIdxRaw - srcRowIdxRaw) !== 1) return;
            // Keep adjacent-only links; auto-flip reversed edges to forward order.
            const forwardColumn =
              tgtRowIdxRaw > srcRowIdxRaw
                ? { source: sourceInfo, target: targetInfo }
                : { source: targetInfo, target: sourceInfo };

            const srcO = cellOrigin(forwardColumn.source.rowName, columnKey, cellStepX, cellStepY);
            const tgtO = cellOrigin(forwardColumn.target.rowName, columnKey, cellStepX, cellStepY);

            const srcIdx = forwardColumn.source.idx;
            const tgtIdx = forwardColumn.target.idx;
            const srcBand = x(srcIdx);
            const tgtBand = x(tgtIdx);
            if (srcBand == null || tgtBand == null) return;

            const sPrev = forwardColumn.source.node.previous ?? 0;
            const sVal = forwardColumn.source.node.value ?? 0;
            const tPrev = forwardColumn.target.node.previous ?? 0;
            const tVal = forwardColumn.target.node.value ?? 0;

            const sBarY = innerH - y(sPrev);
            const sBarH = innerH - y(sVal);
            const tBarY = innerH - y(tPrev);
            const tBarH = innerH - y(tVal);

            const strokeW = linkWidthScale(+edge.value || 0);

            let pathD = "";
            if (!SWAP_AXES) {
              const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + nodeBarOffset + nodeBarWidth / 2;
              const srcCenterY = srcO.y + cellMargin.top + sBarY + Math.max(0, sBarH);
              const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number) + nodeBarOffset + nodeBarWidth / 2;
              const tgtCenterY = tgtO.y + cellMargin.top + tBarY;
              const dy = Math.abs(tgtCenterY - srcCenterY) * 0.5;
              pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX},${srcCenterY + dy}
                       ${tgtCenterX},${tgtCenterY - dy}
                       ${tgtCenterX},${tgtCenterY}`;
            } else {
              const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + nodeBarOffset + nodeBarWidth;
              const srcCenterY = srcO.y + cellMargin.top + (sBarY + Math.max(0, sBarH) / 2);
              const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number) + nodeBarOffset;
              const tgtCenterY = tgtO.y + cellMargin.top + (tBarY + Math.max(0, tBarH) / 2);
              const dx = Math.abs(tgtCenterX - srcCenterX) * 0.5;
              pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX + dx},${srcCenterY}
                       ${tgtCenterX - dx},${tgtCenterY}
                       ${tgtCenterX},${tgtCenterY}`;
            }

            const p = appendLinkPath(linkGroup as any, pathD, edge.value, strokeW, SANKEY_2D_COLUMN_LINK_COLOR, 0.3);
            p.attr("data-column", columnKey);
          });
          });

          // row-links (per epoch/model) -> horizontal inside a row
          linkGroup2 = gRoot.append("g").attr("class", "links row-links");
          Object.keys(data.epoch || {}).forEach((rowKey) => {
          const rowObj = data.epoch[rowKey];
          if (ROW_INDEX.get(rowKey) == null) return;
          const acceptedPairSet = new Set<string>();

          (rowObj.edges || []).forEach((edge: any) => {
            const sLabelRaw = String(edge.source_label ?? "");
            const tLabelRaw = String(edge.target_label ?? "");
            const sColRaw = canonicalLayerName(columnFromLabel(sLabelRaw));
            const tColRaw = canonicalLayerName(columnFromLabel(tLabelRaw));
            const sourceInfo = lookupNode(rowKey, sColRaw, sLabelRaw, edge.source_node_id);
            const targetInfo = lookupNode(rowKey, tColRaw, tLabelRaw, edge.target_node_id);

            if (!sourceInfo || !targetInfo) return;
            const sourceColCanonical = canonicalLayerName(sourceInfo.columnName);
            const targetColCanonical = canonicalLayerName(targetInfo.columnName);
            const srcColIdxRaw = COLUMN_INDEX.get(sourceColCanonical);
            const tgtColIdxRaw = COLUMN_INDEX.get(targetColCanonical);
            if (srcColIdxRaw == null || tgtColIdxRaw == null) return;
            // Strict adjacency only: previous layer <-> next layer.
            if (Math.abs(tgtColIdxRaw - srcColIdxRaw) !== 1) return;
            // Auto-orient reversed data to forward order by column index.
            const forwardRow =
              tgtColIdxRaw > srcColIdxRaw
                ? { source: sourceInfo, target: targetInfo }
                : { source: targetInfo, target: sourceInfo };
            const forwardSrcCol = canonicalLayerName(forwardRow.source.columnName);
            const forwardTgtCol = canonicalLayerName(forwardRow.target.columnName);
            acceptedPairSet.add(`${forwardSrcCol}->${forwardTgtCol}`);

            const srcO = cellOrigin(rowKey, forwardSrcCol, cellStepX, cellStepY);
            const tgtO = cellOrigin(rowKey, forwardTgtCol, cellStepX, cellStepY);

            const srcIdx = forwardRow.source.idx;
            const tgtIdx = forwardRow.target.idx;
            const srcBand = x(srcIdx);
            const tgtBand = x(tgtIdx);
            if (srcBand == null || tgtBand == null) return;

            const sPrev = forwardRow.source.node.previous ?? 0;
            const sVal = forwardRow.source.node.value ?? 0;
            const tPrev = forwardRow.target.node.previous ?? 0;
            const tVal = forwardRow.target.node.value ?? 0;

            const sBarY = innerH - y(sPrev);
            const sBarH = innerH - y(sVal);
            const tBarY = innerH - y(tPrev);
            const tBarH = innerH - y(tVal);

            const strokeW = linkWidthScale(+edge.value || 0);

            let pathD = "";
            if (!SWAP_AXES) {
              const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + nodeBarOffset + nodeBarWidth;
              const srcCenterY = srcO.y + cellMargin.top + (sBarY + Math.max(0, sBarH) / 2);
              const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number) + nodeBarOffset;
              const tgtCenterY = tgtO.y + cellMargin.top + (tBarY + Math.max(0, tBarH) / 2);
              const dx = Math.abs(tgtCenterX - srcCenterX) * 0.5;
              pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX + dx},${srcCenterY}
                       ${tgtCenterX - dx},${tgtCenterY}
                       ${tgtCenterX},${tgtCenterY}`;
            } else {
              const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + nodeBarOffset + nodeBarWidth / 2;
              const srcCenterY = srcO.y + cellMargin.top + sBarY + Math.max(0, sBarH);
              const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number) + nodeBarOffset + nodeBarWidth / 2;
              const tgtCenterY = tgtO.y + cellMargin.top + tBarY;
              const dy = Math.abs(tgtCenterY - srcCenterY) * 0.5;
              pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX},${srcCenterY + dy}
                       ${tgtCenterX},${tgtCenterY - dy}
                       ${tgtCenterX},${tgtCenterY}`;
            }

            const p2 = appendLinkPath(linkGroup2 as any, pathD, edge.value, strokeW, SANKEY_2D_ROW_LINK_COLOR, 0.2);
            p2.attr("data-row", rowKey);
          });

          // Ensure every adjacent layer pair has ordered paths.
          // If raw edges have none for a pair, synthesize a complete conservative bridge
          // from both sides' node masses (cluster-wise), not a single coarse link.
          for (let i = 0; i < COLUMN_NAMES.length - 1; i += 1) {
            const fromCol = COLUMN_NAMES[i];
            const toCol = COLUMN_NAMES[i + 1];
            const pairKey = `${fromCol}->${toCol}`;
            if (acceptedPairSet.has(pairKey)) continue;

            const srcNodes = ((rowObj?.nodes?.[fromCol] || []) as any[]).filter((n) => (n?.value ?? 0) > 0);
            const tgtNodes = ((rowObj?.nodes?.[toCol] || []) as any[]).filter((n) => (n?.value ?? 0) > 0);
            if (srcNodes.length === 0 || tgtNodes.length === 0) continue;

            const bridges = buildGreedyBridgeLinks(srcNodes, tgtNodes);
            const srcO = cellOrigin(rowKey, fromCol, cellStepX, cellStepY);
            const tgtO = cellOrigin(rowKey, toCol, cellStepX, cellStepY);

            bridges.forEach((bridge) => {
              const srcIdx = bridge.sourceIdx;
              const tgtIdx = bridge.targetIdx;
              const srcNode = srcNodes[srcIdx];
              const tgtNode = tgtNodes[tgtIdx];
              const srcBand = x(srcIdx);
              const tgtBand = x(tgtIdx);
              if (srcBand == null || tgtBand == null) return;

              const sPrev = srcNode.previous ?? 0;
              const sVal = srcNode.value ?? 0;
              const tPrev = tgtNode.previous ?? 0;
              const tVal = tgtNode.value ?? 0;
              const sBarY = innerH - y(sPrev);
              const sBarH = innerH - y(sVal);
              const tBarY = innerH - y(tPrev);
              const tBarH = innerH - y(tVal);

              let pathD = "";
              if (!SWAP_AXES) {
                const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + nodeBarOffset + nodeBarWidth;
                const srcCenterY = srcO.y + cellMargin.top + (sBarY + Math.max(0, sBarH) / 2);
                const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number) + nodeBarOffset;
                const tgtCenterY = tgtO.y + cellMargin.top + (tBarY + Math.max(0, tBarH) / 2);
                const dx = Math.abs(tgtCenterX - srcCenterX) * 0.5;
                pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX + dx},${srcCenterY}
                       ${tgtCenterX - dx},${tgtCenterY}
                       ${tgtCenterX},${tgtCenterY}`;
              } else {
                const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + nodeBarOffset + nodeBarWidth / 2;
                const srcCenterY = srcO.y + cellMargin.top + sBarY + Math.max(0, sBarH);
                const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number) + nodeBarOffset + nodeBarWidth / 2;
                const tgtCenterY = tgtO.y + cellMargin.top + tBarY;
                const dy = Math.abs(tgtCenterY - srcCenterY) * 0.5;
                pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX},${srcCenterY + dy}
                       ${tgtCenterX},${tgtCenterY - dy}
                       ${tgtCenterX},${tgtCenterY}`;
              }

              const syntheticW = Math.max(0.8, linkWidthScale(bridge.value));
              const synthetic = appendLinkPath(
                linkGroup2 as any,
                pathD,
                bridge.value,
                syntheticW,
                SANKEY_2D_ROW_LINK_COLOR,
                0.14
              );
              synthetic.attr("data-row", rowKey).attr("data-synthetic", "1");
            });
          }
          });
        }

        // draw nodes after links so they are on top
        if (showBaseSankey) {
          nodeDrawQueue.forEach(({ g, rowName, columnName, nodesForCell }) => {
            g.selectAll("rect.bar")
              .data(nodesForCell)
              .enter()
              .append("rect")
              .attr("class", "bar")
              .attr("x", (_d: any, i: number) => (x(i) as number) + nodeBarOffset)
              .attr("y", (d: any) => innerH - y(d.previous))
              .attr("width", nodeBarWidth)
              .attr("height", (d: any) => innerH - y(d.value))
              .attr("fill", "#aaa")
              .attr("data-row", rowName)
              .attr("data-column", columnName)
              .attr("data-cluster", (d: any) => {
                const c = d?.cluster;
                if (c !== null && c !== undefined && c !== "") return Number(c);
                const label = String(d?.label ?? "");
                const i = label.lastIndexOf("_C");
                if (i === -1) return "";
                return Number(label.slice(i + 2));
              })
              .style("cursor", "pointer")
              .on("click", (event: any, d: any) => {
                event.stopPropagation();
                const c = d?.cluster;
                let cluster = c !== null && c !== undefined && c !== "" ? Number(c) : NaN;
                if (Number.isNaN(cluster)) {
                  const label = String(d?.label ?? "");
                  const i = label.lastIndexOf("_C");
                  if (i !== -1) cluster = Number(label.slice(i + 2));
                }
                if (!Number.isNaN(cluster)) {
                  onNodeSelect?.({ model: rowName, layer: columnName, cluster });
                }
              });
          });
        }

        // slider + mode controls
        const minRange = document.getElementById("minRange") as HTMLInputElement | null;
        const maxRange = document.getElementById("maxRange") as HTMLInputElement | null;
        const minValue = document.getElementById("minValue") as HTMLSpanElement | null;
        const maxValue = document.getElementById("maxValue") as HTMLSpanElement | null;

        if (minRange && maxRange && minValue && maxValue) {
          minRange.min = "0";
          minRange.max = String(maxInstance);
          maxRange.min = "0";
          maxRange.max = String(maxInstance);

          const qMin = urlParams.get("min");
          const qMax = urlParams.get("max");
          if (qMin !== null) minRange.value = String(Math.max(0, Math.min(maxInstance, +qMin)));
          else minRange.value = "0";
          maxRange.value = String(maxInstance);

          minValue.textContent = minRange.value;
          maxValue.textContent = maxRange.value;
          updateSliderTrack();
        }

        function normalizeRange() {
          if (!minRange || !maxRange || !minValue || !maxValue) return [0, maxInstance];
          let a = +minRange.value;
          let b = +maxRange.value;
          if (a > b) {
            const tmp = a;
            a = b;
            b = tmp;
            minRange.value = String(a);
            maxRange.value = String(b);
          }
          minValue.textContent = String(a);
          maxValue.textContent = String(b);
          return [a, b];
        }

        function updateSliderTrack() {
          if (!minRange || !maxRange) return;
          const minV = +minRange.min;
          const maxV = +maxRange.max;
          let a = +minRange.value;
          let b = +maxRange.value;
          if (a > b) {
            const tmp = a;
            a = b;
            b = tmp;
          }
          const startPct = ((a - minV) / (maxV - minV)) * 100;
          const endPct = ((b - minV) / (maxV - minV)) * 100;
          const track = document.getElementById("rangeTrack") as HTMLDivElement | null;
          if (track) {
            track.style.background = `linear-gradient(to right, #ddd ${startPct}%, #9c795b ${startPct}%, #9c795b ${endPct}%, #ddd ${endPct}%)`;
          }
        }

        function updateLinkVisibilityRange(minVal: number, maxVal: number) {
          d3.selectAll("path.link-path").each(function () {
            const el = d3.select(this);
            const v = +el.attr("data-value") || 0;
            if (v < minVal || v > maxVal) el.classed("hidden", true);
            else el.classed("hidden", false);
          });
        }

        normalizeRange();
        updateSliderTrack();
        if (minRange && maxRange) {
          minRange.addEventListener("input", () => {
            const [a, b] = normalizeRange();
            updateLinkVisibilityRange(a, b);
            updateSliderTrack();
          });
          maxRange.addEventListener("input", () => {
            const [a, b] = normalizeRange();
            updateLinkVisibilityRange(a, b);
            updateSliderTrack();
          });
        }

        const btnDefault = document.getElementById("btnDefault") as HTMLButtonElement | null;
        const btnColumn = document.getElementById("btnColumn") as HTMLButtonElement | null;
        const btnRow = document.getElementById("btnRow") as HTMLButtonElement | null;
        const btnSwap = document.getElementById("btnSwap") as HTMLButtonElement | null;

        function setMode(m: "default" | "column" | "row") {
          if (btnDefault && btnColumn && btnRow) {
            btnDefault.style.fontWeight = m === "default" ? "600" : "400";
            btnColumn.style.fontWeight = m === "column" ? "600" : "400";
            btnRow.style.fontWeight = m === "row" ? "600" : "400";
          }

          if (m === "default") {
            d3.selectAll('#chart svg g[id^="cell-"] rect').classed("cell-dimmed", false).classed("cell-highlight", false);
            d3.selectAll('#chart svg g[id^="cell-"] rect.bar').classed("cell-dimmed", false);
            d3.selectAll("path.link-path").classed("link-dimmed", false).classed("link-highlight", false);
          } else if (m === "column") {
            d3.selectAll("path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
            if (!SWAP_AXES) {
              d3.selectAll("g.column-links path.link-path").classed("link-dimmed", false).classed("link-highlight", true);
              d3.selectAll("g.row-links path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
            } else {
              d3.selectAll("g.row-links path.link-path").classed("link-dimmed", false).classed("link-highlight", true);
              d3.selectAll("g.column-links path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
            }
          } else if (m === "row") {
            d3.selectAll("path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
            if (!SWAP_AXES) {
              d3.selectAll("g.row-links path.link-path").classed("link-dimmed", false).classed("link-highlight", true);
              d3.selectAll("g.column-links path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
            } else {
              d3.selectAll("g.column-links path.link-path").classed("link-dimmed", false).classed("link-highlight", true);
              d3.selectAll("g.row-links path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
            }
          }
        }

        btnDefault?.addEventListener("click", () => setMode("default"));
        btnColumn?.addEventListener("click", () => setMode("column"));
        btnRow?.addEventListener("click", () => setMode("row"));
        btnSwap?.addEventListener("click", () => setSwapParam(!SWAP_AXES));

        if (btnSwap) btnSwap.style.fontWeight = SWAP_AXES ? "600" : "400";
        setMode("default");

        function setRowHighlight(rowKey: string, on: boolean) {
          if (on) {
            d3.selectAll('#chart svg g[id^="cell-"] rect, #chart svg g[id^="cell-"] rect.bar').classed("cell-dimmed", true).classed("cell-highlight", false);
            d3.selectAll("path.link-path").classed("link-dimmed", true).classed("link-highlight", false);

            X_NAMES.forEach((xKey) => {
              const rowName = SWAP_AXES ? xKey : rowKey;
              const columnName = SWAP_AXES ? rowKey : xKey;
              const cellSel = d3.select(`#cell-${rowName}-${columnName}`);
              if (!cellSel.empty()) {
                cellSel.select("rect").classed("cell-dimmed", false).classed("cell-highlight", true);
                cellSel.selectAll("rect.bar").classed("cell-dimmed", false);
              }
            });

            if (!SWAP_AXES) {
              d3.selectAll("g.row-links path.link-path").each(function () {
                const el = d3.select(this);
                if (el.attr("data-row") === rowKey) el.classed("link-dimmed", false).classed("link-highlight", true);
              });
            } else {
              d3.selectAll("g.column-links path.link-path").each(function () {
                const el = d3.select(this);
                if (el.attr("data-column") === rowKey) el.classed("link-dimmed", false).classed("link-highlight", true);
              });
            }
          } else {
            d3.selectAll('#chart svg g[id^="cell-"] rect').classed("cell-dimmed", false).classed("cell-highlight", false);
            d3.selectAll('#chart svg g[id^="cell-"] rect.bar').classed("cell-dimmed", false);
            d3.selectAll("path.link-path").classed("link-dimmed", false).classed("link-highlight", false);
          }
        }

        function setColHighlight(colKey: string, on: boolean) {
          if (on) {
            d3.selectAll('#chart svg g[id^="cell-"] rect, #chart svg g[id^="cell-"] rect.bar').classed("cell-dimmed", true).classed("cell-highlight", false);
            d3.selectAll("path.link-path").classed("link-dimmed", true).classed("link-highlight", false);

            Y_NAMES.forEach((yKey) => {
              const rowName = SWAP_AXES ? colKey : yKey;
              const columnName = SWAP_AXES ? yKey : colKey;
              const cellSel = d3.select(`#cell-${rowName}-${columnName}`);
              if (!cellSel.empty()) {
                cellSel.select("rect").classed("cell-dimmed", false).classed("cell-highlight", true);
                cellSel.selectAll("rect.bar").classed("cell-dimmed", false);
              }
            });

            if (!SWAP_AXES) {
              d3.selectAll("g.column-links path.link-path").each(function () {
                const el = d3.select(this);
                if (el.attr("data-column") === colKey) el.classed("link-dimmed", false).classed("link-highlight", true);
              });
            } else {
              d3.selectAll("g.row-links path.link-path").each(function () {
                const el = d3.select(this);
                if (el.attr("data-row") === colKey) el.classed("link-dimmed", false).classed("link-highlight", true);
              });
            }
          } else {
            d3.selectAll('#chart svg g[id^="cell-"] rect').classed("cell-dimmed", false).classed("cell-highlight", false);
            d3.selectAll('#chart svg g[id^="cell-"] rect.bar').classed("cell-dimmed", false);
            d3.selectAll("path.link-path").classed("link-dimmed", false).classed("link-highlight", false);
          }
        }

        function setCellHighlight(rowName: string, columnName: string, on: boolean) {
          if (on) {
            d3.selectAll('#chart svg g[id^="cell-"] rect, #chart svg g[id^="cell-"] rect.bar').classed("cell-dimmed", true).classed("cell-highlight", false);
            d3.selectAll("path.link-path").classed("link-dimmed", true).classed("link-highlight", false);

            ROW_NAMES.forEach((row) => {
              const cellSel = d3.select(`#cell-${row}-${columnName}`);
              if (!cellSel.empty()) {
                cellSel.select("rect").classed("cell-dimmed", false).classed("cell-highlight", true);
                cellSel.selectAll("rect.bar").classed("cell-dimmed", false);
              }
            });

            COLUMN_NAMES.forEach((column) => {
              const cellSel = d3.select(`#cell-${rowName}-${column}`);
              if (!cellSel.empty()) {
                cellSel.select("rect").classed("cell-dimmed", false).classed("cell-highlight", true);
                cellSel.selectAll("rect.bar").classed("cell-dimmed", false);
              }
            });

            d3.selectAll("g.row-links path.link-path").each(function () {
              const el = d3.select(this);
              if (el.attr("data-row") === rowName) el.classed("link-dimmed", false).classed("link-highlight", true);
            });

            d3.selectAll("g.column-links path.link-path").each(function () {
              const el = d3.select(this);
              if (el.attr("data-column") === columnName) el.classed("link-dimmed", false).classed("link-highlight", true);
            });
          } else {
            d3.selectAll('#chart svg g[id^="cell-"] rect').classed("cell-dimmed", false).classed("cell-highlight", false);
            d3.selectAll('#chart svg g[id^="cell-"] rect.bar').classed("cell-dimmed", false);
            d3.selectAll("path.link-path").classed("link-dimmed", false).classed("link-highlight", false);
          }
        }

        // legends moved outside zoomed canvas
      })
      .catch((err) => {
        console.error("Failed to build chart:", err);
        container.innerHTML = "<div style='color: red'>Failed to load visualization data.</div>";
      });
  }, [dataPath, transitionStatsPath, showBaseSankey, showCellBackground, showGapHeatmap]);

  useEffect(() => {
    const container = chartRef.current;
    if (!container) return;
    const sel = d3.select(container).selectAll<SVGRectElement, any>("rect.bar");
    const focusModel = selectedNode?.model ?? activeCell?.model ?? null;
    const focusLayer = selectedNode?.layer ?? activeCell?.layer ?? null;
    if (!focusModel || !focusLayer) {
      sel.attr("fill", "#aaa");
      sel.classed("node-highlight", false).classed("node-dimmed", false);
      // Keep current cell/link highlight state (e.g., locked 2D cell focus) when cluster selection is cleared.
      return;
    }
    const clusterValues: number[] = [];
    sel.each(function () {
      const el = d3.select(this);
      const row = el.attr("data-row");
      const col = el.attr("data-column");
      if (row !== focusModel || col !== focusLayer) return;
      const cluster = Number(el.attr("data-cluster"));
      if (Number.isFinite(cluster)) clusterValues.push(cluster);
    });
    const maxCluster = clusterValues.length ? Math.max(...clusterValues) : 0;
    const color = d3
      .scaleOrdinal<number, string>()
      .domain(d3.range(maxCluster + 1))
      .range(CLUSTER_COLORS);
    sel.each(function () {
      const el = d3.select(this);
      const row = el.attr("data-row");
      const col = el.attr("data-column");
      const cluster = Number(el.attr("data-cluster"));
      if (row === focusModel && col === focusLayer && Number.isFinite(cluster)) {
        el.attr("fill", color(Math.max(0, cluster)));
      } else {
        el.attr("fill", "#aaa");
      }
      if (!selectedNode) {
        el.classed("node-highlight", false);
        el.classed("node-dimmed", false);
        return;
      }
      const isMatch =
        row === selectedNode.model &&
        col === selectedNode.layer &&
        Number.isFinite(cluster) &&
        cluster === selectedNode.cluster;
      el.classed("node-highlight", isMatch);
      el.classed("node-dimmed", false);
    });

    if (!selectedNode) return;

    const rowKey = selectedNode.model;
    const colKey = selectedNode.layer;
    const cellSel = d3.select(container).selectAll('#chart svg g[id^="cell-"] rect, #chart svg g[id^="cell-"] rect.bar');
    cellSel.classed("cell-dimmed", true).classed("cell-highlight", false);

    d3.select(container)
      .selectAll<SVGGElement, any>('#chart svg g[id^="cell-"]')
      .each(function () {
        const g = d3.select(this);
        const id = g.attr("id") || "";
        if (!id.startsWith("cell-")) return;
        const raw = id.slice(5);
        const idx = raw.lastIndexOf("-");
        if (idx === -1) return;
        const row = raw.slice(0, idx);
        const col = raw.slice(idx + 1);
        if (row === rowKey || col === colKey) {
          g.select("rect").classed("cell-dimmed", false).classed("cell-highlight", true);
          g.selectAll("rect.bar").classed("cell-dimmed", false);
        }
      });

    d3.select(container)
      .selectAll("path.link-path")
      .classed("link-dimmed", true)
      .classed("link-highlight", false);
    d3.select(container)
      .selectAll("g.row-links path.link-path")
      .each(function () {
        const el = d3.select(this);
        if (el.attr("data-row") === rowKey) el.classed("link-dimmed", false).classed("link-highlight", true);
      });
    d3.select(container)
      .selectAll("g.column-links path.link-path")
      .each(function () {
        const el = d3.select(this);
        if (el.attr("data-column") === colKey) el.classed("link-dimmed", false).classed("link-highlight", true);
      });
  }, [selectedNode, activeCell]);

  useEffect(() => {
    if (!SHOW_FMI_CSPA) return;
    fetch(metricsPath)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load avg fmi/cspa metrics");
        return res.json();
      })
      .then((json) => {
        const layers: Array<{ layer: string; avg_fmi: number; avg_cspa: number }> = json?.layers || [];
        const models: Array<{ model: string; avg_fmi: number; avg_cspa: number }> = json?.models || [];
        const next = layers
          .map((d) => ({
            layer: d.layer,
            fmi: Number(d.avg_fmi),
            cspa: Number(d.avg_cspa),
          }))
          .filter((d) => Number.isFinite(d.fmi) && Number.isFinite(d.cspa));
        const nextModels = models
          .map((d) => ({
            model: d.model,
            fmi: Number(d.avg_fmi),
            cspa: Number(d.avg_cspa),
          }))
          .filter((d) => Number.isFinite(d.fmi) && Number.isFinite(d.cspa));
        setFmiData(next);
        setRowMetricData(nextModels);
      })
      .catch((err) => {
        console.error(err);
      });
  }, [metricsPath]);

  useEffect(() => {
    if (!SHOW_FMI_CSPA) {
      const container = fmiRef.current;
      if (container) container.innerHTML = "";
      return;
    }
    const container = fmiRef.current;
    if (!container) return;
    if (!layout || !fmiData.length) {
      container.innerHTML = "";
      return;
    }

    const ro = new ResizeObserver(() => draw());
    ro.observe(container);

    function draw() {
      if (!container || !layout) return;
      container.innerHTML = "";
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(30, container.clientHeight || 40);
      const margin = { top: 6, right: 8, bottom: 6, left: 0 };
      const innerH = Math.max(1, height - margin.top - margin.bottom);

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      const fmiByLayer = new Map(fmiData.map((d) => [d.layer, d.fmi]));
      const cspaByLayer = new Map(fmiData.map((d) => [d.layer, d.cspa]));
      const fmiValues = fmiData.map((d) => d.fmi);
      const cspaValues = fmiData.map((d) => d.cspa);
      const fmiMin = Math.min(...fmiValues);
      const fmiMax = Math.max(...fmiValues);
      const fmiColor = d3
        .scaleLinear<string>()
        .domain([fmiMin, fmiMin + (fmiMax - fmiMin) * 0.4, fmiMin + (fmiMax - fmiMin) * 0.8, fmiMax])
        .range(["#f0f0f0", "#f0f0f0", "#cda67f", "#9c795b"]);
      const cspaMin = Math.min(...cspaValues);
      const cspaMax = Math.max(...cspaValues);
      const cspaColor = d3
        .scaleLinear<string>()
        .domain([cspaMin, cspaMin + (cspaMax - cspaMin) * 0.4, cspaMin + (cspaMax - cspaMin) * 0.8, cspaMax])
        .range(["#f0f0f0", "#f0f0f0", "#cda67f", "#9c795b"]);

      const points = layout.colNames
        .map((name, i) => {
          const x =
            transform.x +
            transform.scale *
              (layout.margin.left + i * layout.cellStepX + layout.cellMargin.left + layout.innerW / 2);
          const fmi = fmiByLayer.get(name);
          const cspa = cspaByLayer.get(name);
          if (fmi == null || cspa == null) return null;
          return { x, fmi, cspa };
        })
        .filter((d): d is { x: number; fmi: number; cspa: number } => !!d);

      const cellW = transform.scale * layout.cellW;
      const rowHeight = Math.max(6, innerH * 0.28);
      const rowGap = 6;
      const fmiY = 16;
      const cspaY = fmiY + rowHeight + rowGap;

      g.append("text")
        .attr("x", -8 + 30)
        .attr("y", fmiY + rowHeight / 2 + 2)
        .attr("text-anchor", "end")
        .attr("font-size", 11)
        .attr("fill", "#64748b")
        .text("FMI");
      g.append("text")
        .attr("x", -8 + 30)
        .attr("y", cspaY + rowHeight / 2 + 2)
        .attr("text-anchor", "end")
        .attr("font-size", 11)
        .attr("fill", "#64748b")
        .text("CSPA");

      g.selectAll("rect.fmi-cell")
        .data(points)
        .enter()
        .append("rect")
        .attr("class", "fmi-cell")
        .attr("x", (d) => (d.x - margin.left) - cellW / 2 + 40)
        .attr("y", fmiY)
        .attr("width", Math.max(1, cellW - 2))
        .attr("height", rowHeight)
        .attr("rx", 2)
        .attr("fill", (d) => fmiColor(d.fmi));

      g.selectAll("text.fmi-val")
        .data(points)
        .enter()
        .append("text")
        .attr("class", "fmi-val")
        .attr("x", (d) => d.x - margin.left + 40)
        .attr("y", fmiY + rowHeight / 2 + 3)
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("fill", "#0f172a")
        .text((d) => d.fmi.toFixed(3));

      g.selectAll("rect.cspa-cell")
        .data(points)
        .enter()
        .append("rect")
        .attr("class", "cspa-cell")
        .attr("x", (d) => (d.x - margin.left) - cellW / 2 + 40)
        .attr("y", cspaY)
        .attr("width", Math.max(1, cellW - 2))
        .attr("height", rowHeight)
        .attr("rx", 2)
        .attr("fill", (d) => cspaColor(d.cspa));

      g.selectAll("text.cspa-val")
        .data(points)
        .enter()
        .append("text")
        .attr("class", "cspa-val")
        .attr("x", (d) => d.x - margin.left + 40)
        .attr("y", cspaY + rowHeight / 2 + 3)
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("fill", "#0f172a")
        .text((d) => d.cspa.toFixed(3));
    }

    draw();
    return () => ro.disconnect();
  }, [layout, transform, fmiData]);

  useEffect(() => {
    if (!SHOW_FMI_CSPA) {
      const container = rowMetricRef.current;
      if (container) container.innerHTML = "";
      return;
    }
    const container = rowMetricRef.current;
    if (!container) return;
    if (!layout || !rowMetricData.length) {
      container.innerHTML = "";
      return;
    }

    const ro = new ResizeObserver(() => draw());
    ro.observe(container);

    function draw() {
      if (!container || !layout) return;
      container.innerHTML = "";
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(30, container.clientHeight || 40);
      const margin = { top: 6, right: 6, bottom: 6, left: 6 };
      const innerW = Math.max(1, width - margin.left - margin.right);

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      const fmiByModel = new Map(rowMetricData.map((d) => [d.model, d.fmi]));
      const cspaByModel = new Map(rowMetricData.map((d) => [d.model, d.cspa]));
      const fmiValues = rowMetricData.map((d) => d.fmi);
      const cspaValues = rowMetricData.map((d) => d.cspa);
      const fmiMin = Math.min(...fmiValues);
      const fmiMax = Math.max(...fmiValues);
      const cspaMin = Math.min(...cspaValues);
      const cspaMax = Math.max(...cspaValues);

      const fmiColor = d3
        .scaleLinear<string>()
        .domain([fmiMin, fmiMin + (fmiMax - fmiMin) * 0.4, fmiMin + (fmiMax - fmiMin) * 0.8, fmiMax])
        .range(["#f0f0f0", "#f0f0f0", "#cda67f", "#9c795b"]);
      const cspaColor = d3
        .scaleLinear<string>()
        .domain([cspaMin, cspaMin + (cspaMax - cspaMin) * 0.4, cspaMin + (cspaMax - cspaMin) * 0.8, cspaMax])
        .range(["#f0f0f0", "#f0f0f0", "#cda67f", "#9c795b"]);

      const rowHeight = Math.max(10, transform.scale * layout.cellH);
      const blockH = rowHeight;
      const gap = 2;
      const blockW = Math.min(15, Math.max(6, Math.floor(innerW / 2) - gap));
      const fmiX = 8;
      const cspaX = Math.min(innerW - blockW, blockW + gap) + 8;
      const labelY = 6;

      layout.rowNames.forEach((name, i) => {
        const y =
          transform.y +
          transform.scale * (layout.margin.top + i * layout.cellStepY + layout.cellMargin.top + layout.innerH / 2);
        const fmi = fmiByModel.get(name);
        const cspa = cspaByModel.get(name);
        if (fmi == null || cspa == null) return;

        const yTop = y - margin.top - blockH / 2 + 7;
        g.append("rect")
          .attr("x", fmiX)
          .attr("y", yTop)
          .attr("width", blockW)
          .attr("height", blockH)
          .attr("rx", 2)
          .attr("fill", fmiColor(fmi));

        g.append("rect")
          .attr("x", cspaX )
          .attr("y", yTop )
          .attr("width", blockW)
          .attr("height", blockH)
          .attr("rx", 2)
          .attr("fill", cspaColor(cspa));

        g.append("text")
          .attr("x", fmiX + blockW / 2)
          .attr("y", yTop + blockH / 2 + 2)
          .attr("text-anchor", "middle")
          .attr("font-size", 9)
          .attr("fill", "#111827")
          .attr("transform", `rotate(-90, ${fmiX + blockW / 2}, ${yTop + blockH / 2})`)
          .text(fmi.toFixed(3));

        g.append("text")
          .attr("x", cspaX + blockW / 2)
          .attr("y", yTop + blockH / 2 + 2)
          .attr("text-anchor", "middle")
          .attr("font-size", 9)
          .attr("fill", "#111827")
          .attr("transform", `rotate(-90, ${cspaX + blockW / 2}, ${yTop + blockH / 2})`)
          .text(cspa.toFixed(3));
      });

      g.append("text")
        .attr("x", fmiX + blockW / 2 + 5)
        .attr("y", labelY + 2)
        .attr("text-anchor", "middle")
        .attr("font-size", 10)
        .attr("fill", "#6b7280")
        .attr("transform", `rotate(-90, ${fmiX + blockW / 2}, ${labelY})`)
        .text("FMI");

      g.append("text")
        .attr("x", cspaX + blockW / 2 + 5)
        .attr("y", labelY+2)
        .attr("text-anchor", "middle")
        .attr("font-size", 8)
        .attr("fill", "#6b7280")
        .attr("transform", `rotate(-90, ${cspaX + blockW / 2}, ${labelY})`)
        .text("CSPA");
    }

    draw();
    return () => ro.disconnect();
  }, [layout, transform, rowMetricData]);

  return (
    <div className="timeline-root h-full flex flex-col" style={{ marginLeft: 0 }}>
      <div id="controls" className="shrink-0">
        <span style={{ display: "inline-flex", alignItems: "center", marginLeft: 5, marginRight: 8, gap: 6 }}>
          <label>Value range:</label>
          <span
            className="value-range-help"
            data-tooltip="Value range is a link-value filter for the 2D Sankey view. It controls which path segments are visible: only links with values between the selected minimum and maximum are displayed, and links outside that range are hidden."
            aria-label="Value range help"
          >
            ?
          </span>
        </span>
        <span id="minValue" style={{ marginRight: 8 }}>0</span>
        <div className="range-slider" id="rangeSlider">
          <div className="range-track" id="rangeTrack"></div>
          <input id="minRange" className="range-min" type="range" min={0} max={maxInstanceValue} step={1} defaultValue={0} />
          <input id="maxRange" className="range-max" type="range" min={0} max={maxInstanceValue} step={1} defaultValue={maxInstanceValue} />
        </div>
        <span id="maxValue" style={{ marginLeft: 8 }}>{maxInstanceValue}</span>
        <div id="controlButtons" style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center" }}>
          <button id="btnDefault">Default</button>
          <button id="btnColumn">Column</button>
          <button id="btnRow">Row</button>
          <button id="btnSwap">Swap X/Y</button>
          <button
            type="button"
            onClick={() => setShowBaseSankey((v) => !v)}
            style={{ fontWeight: showBaseSankey ? 600 : 400 }}
          >
            Sankey
          </button>
          <button
            type="button"
            onClick={() => setShowCellBackground((v) => !v)}
            style={{ fontWeight: showCellBackground ? 600 : 400 }}
          >
            Cell BG
          </button>
          <button
            type="button"
            onClick={() => setShowGapHeatmap((v) => !v)}
            style={{ fontWeight: showGapHeatmap ? 600 : 400 }}
          >
            Gap Heat
          </button>
        </div>
      </div>
      {/* FMI/CSPA panel is temporarily hidden, keep code for future use.
      <div id="fmiChart" ref={fmiRef} style={{ width: "100%", height: 78, marginTop: 4, marginBottom: 6, marginLeft: 0 }} />
      */}

      <div className="w-full flex-1 min-h-0 overflow-hidden">
        <div className="h-full flex">
          {/* FMI/CSPA row metric strip is temporarily hidden, keep code for future use.
          <div id="rowMetricChart" ref={rowMetricRef} className="shrink-0 w-10 h-full pointer-events-none" />
          */}
          <div className="relative flex-1 min-w-0 pl-8">
            <ZoomPan
              className="w-full h-full"
              panPadding={220}
              externalTransform={externalTransform}
              onTransform={(t) => {
                setTransform(t);
                onTransform?.(t);
              }}
            >
              <div id="chart" ref={chartRef}></div>
            </ZoomPan>

            {layout && (
              <>
                <div className="absolute left-0 top-0 right-0 h-5 pointer-events-none bg-white/90 z-10 overflow-hidden">
                  {layout.colNames.map((name, i) => (
                    <div
                      key={name}
                      className="absolute top-0 h-5 text-sm text-zinc-700 text-center whitespace-nowrap"
                      style={{
                        left:
                          transform.x +
                          transform.scale *
                            (layout.margin.left + i * layout.cellStepX + layout.cellMargin.left + layout.innerW / 2) +
                          24,
                        transform: "translateX(-50%)",
                      }}
                    >
                      {toShortLayerName(name)}
                    </div>
                  ))}
                </div>
                <div className="absolute left-0 top-0 bottom-0 w-10 pointer-events-none bg-white/90 z-10 overflow-hidden">
                  {layout.rowNames.map((name, i) => (
                    <div
                      key={name}
                      className="absolute left-[2px] text-sm text-zinc-700 whitespace-nowrap"
                      style={{
                        top:
                          transform.y +
                          transform.scale *
                            (layout.margin.top + i * layout.cellStepY + layout.cellMargin.top + layout.innerH / 2),
                        transform: "translateY(-50%)",
                        writingMode: "vertical-lr",
                        textOrientation: "mixed",
                      }}
                    >
                      {toShortLayerName(name)}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <style jsx>{`
        .timeline-root {
          margin: 0;
          font-family: system-ui, -apple-system, BlinkMacSystemFont;
          background: #fff;
        }

        #controls {
          width: 100%;
          max-width: 100%;
          margin: 8px 0;
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 16px;
          flex-wrap: wrap;
          position: relative;
          padding-right: 260px;
        }

        #chart {
          width: fit-content;
          max-width: none;
          margin: 0;
        }

        input[type="range"] {
          width: 360px;
        }

        :global(.link-path.hidden) {
          display: none;
        }

        .range-slider {
          position: relative;
          width: 360px;
          height: 36px;
          display: inline-block;
          align-items: center;
        }

        #controlButtons {
          position: absolute;
          right: 8px;
          top: 0;
        }

        .range-slider input[type="range"] {
          position: absolute;
          left: 0;
          top: 12px;
          width: 100%;
          pointer-events: none;
          -webkit-appearance: none;
          background: transparent;
        }

        .range-slider input.range-min {
          z-index: 3;
        }

        .range-slider input.range-max {
          z-index: 2;
        }

        .range-slider input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          pointer-events: auto;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: #fff;
          border: 2px solid #666;
          box-shadow: 0 0 2px rgba(0, 0, 0, 0.3);
        }

        .range-slider input[type="range"]::-moz-range-thumb {
          pointer-events: auto;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: #fff;
          border: 2px solid #666;
          box-shadow: 0 0 2px rgba(0, 0, 0, 0.3);
        }

        .range-track {
          position: absolute;
          left: 0;
          top: 18px;
          height: 6px;
          width: 100%;
          background: linear-gradient(to right, #ddd 0%, #ddd 100%);
          border-radius: 3px;
        }

        .value-range-help {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 14px;
          height: 14px;
          border-radius: 999px;
          border: 1px solid #9ca3af;
          color: #4b5563;
          font-size: 10px;
          line-height: 1;
          cursor: default;
          user-select: none;
          position: relative;
        }

        .value-range-help::after {
          content: attr(data-tooltip);
          position: absolute;
          left: 6px;
          top: calc(100% + 14px);
          transform: none;
          width: 280px;
          max-width: min(280px, calc(100vw - 24px));
          padding: 8px 10px;
          border-radius: 6px;
          background: rgba(17, 24, 39, 0.95);
          color: #fff;
          font-size: 11px;
          line-height: 1.35;
          white-space: normal;
          text-align: left;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
          opacity: 0;
          visibility: hidden;
          pointer-events: none;
          transition: opacity 0.12s ease;
          z-index: 30;
        }

        .value-range-help::before {
          content: "";
          position: absolute;
          left: 8px;
          top: calc(100% + 9px);
          transform: none;
          border: 5px solid transparent;
          border-bottom-color: rgba(17, 24, 39, 0.95);
          opacity: 0;
          visibility: hidden;
          pointer-events: none;
          transition: opacity 0.12s ease;
          z-index: 30;
        }

        .value-range-help:hover::after,
        .value-range-help:hover::before {
          opacity: 1;
          visibility: visible;
        }

        :global(.cell-dimmed) {
          opacity: 0.12 !important;
        }

        :global(.link-dimmed) {
          stroke-opacity: 0.06 !important;
        }

        :global(#chart svg rect.bar.node-highlight) {
          stroke: #ef4444 !important;
          stroke-width: 2 !important;
        }

        :global(#chart svg rect.bar.node-dimmed) {
          opacity: 0.18 !important;
        }

        :global(#chart svg path.link-path) {
          pointer-events: none;
        }
      `}</style>
    </div>
  );
}
