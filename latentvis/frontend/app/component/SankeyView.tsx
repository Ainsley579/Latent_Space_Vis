"use client";
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import ZoomPan from "./ZoomPan";
import { CLUSTER_COLORS, SANKEY_2D_COLUMN_LINK_COLOR, SANKEY_2D_ROW_LINK_COLOR } from "../theme/colors";

type CellInfoTag = {
    rowName: string;
    columnName: string;
    epochName: string;
    layerName: string;
    col: number;
    row: number;
};

type HoverTooltip = {
    x: number;
    y: number;
    title: string;
    lines: string[];
};

function parseClusterFromLabel(label: string): number {
    const m = /_C(\d+)$/.exec(label);
    return m ? Number(m[1]) : 0;
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
    data: any;
    selectedSplits?: string[];
    showFfn2?: boolean;
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
    reverseEpochOrder?: boolean;
    onToggleEpochOrder?: () => void;
    onVisibleEpochsChange?: (epochs: string[]) => void;
    cellMetricByCell?: Record<string, Record<string, number | null>> | null;
    selectedCellMetric?: string;
    renderRequestId?: number;
    onRenderComplete?: (requestId: number) => void;
};

type SankeyRenderNode = {
    node_id: number;
    epoch: string;
    layer: string;
    epoch_idx: number;
    layer_idx: number;
    cell_index: number;
    cluster_id: number;
    member_count: number;
    value: number;
    previous: number;
    split_counts: {
        train: number;
        test: number;
        val: number;
    };
};

type SankeyRenderData = {
    epochs: string[];
    layers: string[];
    molecule_ids: number[];
    cells: Array<{ children: number[] }>;
    nodes: SankeyRenderNode[];
    connectivity: Array<Array<[number, number]>>;
    epoch?: Record<string, any>;
    layer?: Record<string, any>;
};

function buildRenderData(input: any, selectedSplits?: string[]): SankeyRenderData | null {
    if (!input || !Array.isArray(input?.epochs) || !Array.isArray(input?.layers)) return null;
    // Legacy shape already carries children arrays for cells.
    if (
        Array.isArray(input?.cells) &&
        input.cells.length > 0 &&
        Array.isArray(input.cells[0]?.children) &&
        Array.isArray(input?.nodes) &&
        Array.isArray(input?.connectivity)
    ) {
        return input as SankeyRenderData;
    }

    // Compact shape adapter: aggregate per (epoch_idx, layer_idx, cluster_id) across splits.
    const epochs: string[] = input.epochs.map((x: any) => String(x));
    const layers: string[] = input.layers.map((x: any) => String(x));
    const moleculeIds: number[] = Array.isArray(input?.molecule_ids) ? input.molecule_ids.map((x: any) => Number(x) || 0) : [];
    const originalNodes: any[] = Array.isArray(input?.nodes) ? input.nodes : [];
    const originalConnectivity: Array<Array<[number, number]>> = Array.isArray(input?.connectivity) ? input.connectivity : [];
    const splitNames: string[] = Array.isArray(input?.split_names)
        ? input.split_names.map((x: any) => String(x).toLowerCase())
        : [];
    const splitConnectivity: Array<Array<Array<[number, number]>>> = Array.isArray(input?.split_connectivity)
        ? input.split_connectivity
        : [];
    const normalizedSelectedSplits = Array.isArray(selectedSplits)
        ? selectedSplits.map((x) => String(x).toLowerCase())
        : [];
    const selectedSplitSet =
        normalizedSelectedSplits.length > 0 ? new Set(normalizedSelectedSplits) : null;
    const allowedSplitIndices = new Set<number>();
    if (splitNames.length > 0) {
        splitNames.forEach((name, idx) => {
            if (!selectedSplitSet || selectedSplitSet.has(name)) allowedSplitIndices.add(idx);
        });
    }
    const nCells = epochs.length * layers.length;

    const aggKeyToId = new Map<string, number>();
    const aggNodes: SankeyRenderNode[] = [];
    const originalToAgg = new Map<number, number>();

    function keyOf(epochIdx: number, layerIdx: number, clusterId: number): string {
        return `${epochIdx}|${layerIdx}|${clusterId}`;
    }

    originalNodes.forEach((n: any, idx: number) => {
        const nodeSplit = Number(n?.split);
        if (
            allowedSplitIndices.size > 0 &&
            Number.isFinite(nodeSplit) &&
            !allowedSplitIndices.has(nodeSplit)
        ) {
            return;
        }
        const epochIdx = Number(n?.epoch_idx ?? -1);
        const layerIdx = Number(n?.layer_idx ?? -1);
        const clusterId = Number(n?.cluster_id ?? -1);
        const memberCount = Number(n?.member_count ?? 0);
        if (!Number.isFinite(epochIdx) || !Number.isFinite(layerIdx) || !Number.isFinite(clusterId)) return;
        if (epochIdx < 0 || epochIdx >= epochs.length || layerIdx < 0 || layerIdx >= layers.length) return;
        const key = keyOf(epochIdx, layerIdx, clusterId);
        let aggId = aggKeyToId.get(key);
        if (aggId == null) {
            aggId = aggNodes.length;
            aggKeyToId.set(key, aggId);
            aggNodes.push({
                node_id: aggId,
                epoch: epochs[epochIdx],
                layer: layers[layerIdx],
                epoch_idx: epochIdx,
                layer_idx: layerIdx,
                cell_index: layerIdx + epochIdx * layers.length,
                cluster_id: clusterId,
                member_count: 0,
                value: 0,
                previous: 0,
                split_counts: { train: 0, test: 0, val: 0 },
            });
        }
        aggNodes[aggId].member_count += Math.max(0, memberCount);
        aggNodes[aggId].value = aggNodes[aggId].member_count;
        const splitName =
            Number.isFinite(nodeSplit) && nodeSplit >= 0 && nodeSplit < splitNames.length
                ? String(splitNames[nodeSplit]).toLowerCase()
                : "";
        if (splitName === "train") aggNodes[aggId].split_counts.train += Math.max(0, memberCount);
        else if (splitName === "test") aggNodes[aggId].split_counts.test += Math.max(0, memberCount);
        else if (splitName === "val") aggNodes[aggId].split_counts.val += Math.max(0, memberCount);
        originalToAgg.set(idx, aggId);
    });

    const byCell = new Map<number, SankeyRenderNode[]>();
    aggNodes.forEach((n) => {
        const arr = byCell.get(n.cell_index) ?? [];
        arr.push(n);
        byCell.set(n.cell_index, arr);
    });
    byCell.forEach((arr) => {
        arr.sort((a, b) => a.cluster_id - b.cluster_id);
        let prev = 0;
        arr.forEach((n) => {
            n.previous = prev;
            prev += n.member_count;
        });
    });

    const cells: Array<{ children: number[] }> = Array.from({ length: nCells }, () => ({ children: [] }));
    byCell.forEach((arr, cellIndex) => {
        if (cellIndex < 0 || cellIndex >= nCells) return;
        cells[cellIndex] = { children: arr.map((n) => n.node_id) };
    });

    const connectivitySource: Array<Array<[number, number]>> = (() => {
        if (!Array.isArray(splitConnectivity) || splitConnectivity.length === 0) return originalConnectivity;
        const acc: Array<Array<[number, number]>> = Array.from({ length: originalNodes.length }, () => []);
        const edgeMapBySource = new Map<string, number>();
        splitConnectivity.forEach((splitEdges, splitIdx) => {
            if (allowedSplitIndices.size > 0 && !allowedSplitIndices.has(splitIdx)) return;
            if (!Array.isArray(splitEdges)) return;
            splitEdges.forEach((targets: any, sourceOrigId: number) => {
                if (!Array.isArray(targets)) return;
                targets.forEach((edge: any) => {
                    const targetOrigId = Number(edge?.[0]);
                    const count = Number(edge?.[1]);
                    if (!Number.isFinite(targetOrigId) || !Number.isFinite(count) || count <= 0) return;
                    const key = `${sourceOrigId}|${targetOrigId}`;
                    edgeMapBySource.set(key, (edgeMapBySource.get(key) ?? 0) + count);
                });
            });
        });
        edgeMapBySource.forEach((count, key) => {
            const [src, dst] = key.split("|").map((x) => Number(x));
            if (!Number.isFinite(src) || !Number.isFinite(dst)) return;
            if (src < 0 || src >= acc.length) return;
            acc[src].push([dst, count]);
        });
        return acc;
    })();

    const edgeMap = new Map<string, number>();
    connectivitySource.forEach((targets, sourceOrigId) => {
        const sourceAggId = originalToAgg.get(sourceOrigId);
        if (sourceAggId == null || !Array.isArray(targets)) return;
        targets.forEach((edge: any) => {
            const targetOrigId = Number(edge?.[0]);
            const count = Number(edge?.[1]);
            if (!Number.isFinite(targetOrigId) || !Number.isFinite(count) || count <= 0) return;
            const targetAggId = originalToAgg.get(targetOrigId);
            if (targetAggId == null) return;
            const k = `${sourceAggId}|${targetAggId}`;
            edgeMap.set(k, (edgeMap.get(k) ?? 0) + count);
        });
    });

    const connectivity: Array<Array<[number, number]>> = Array.from({ length: aggNodes.length }, () => []);
    edgeMap.forEach((count, key) => {
        const [s, t] = key.split("|").map((x) => Number(x));
        if (!Number.isFinite(s) || !Number.isFinite(t)) return;
        if (s < 0 || s >= connectivity.length) return;
        connectivity[s].push([t, count]);
    });
    connectivity.forEach((list) => list.sort((a, b) => a[0] - b[0]));

    return {
        epochs,
        layers,
        molecule_ids: moleculeIds,
        cells,
        nodes: aggNodes,
        connectivity,
        epoch: input?.epoch ?? {},
        layer: input?.layer ?? {},
    };
}


const margin = { top: 20, left: 0, right: 0, bottom: 20 };
const CELL_W = 120;
const CELL_H = 60;
const CELL_GAP = 10;
const cellStepX = CELL_W + CELL_GAP;
const cellStepY = CELL_H + CELL_GAP;
const MAX_NODE_BAR_WIDTH = 20;
const EPOCH_THRESHOLD_MAX = 100;


export default function SankeyView({
    data,
    selectedSplits,
    showFfn2 = false,
    transitionStatsPath = "/data/dimenet/clustering_metrics/layer_transition_by_epoch.json",
    onModelSelect,
    onTransform,
    externalTransform,
    onLayerSelect,
    onNodeSelect,
    selectedNode,
    activeCell,
    onCellSelect,
    onSwapAxes,
    reverseEpochOrder = false,
    onToggleEpochOrder,
    onVisibleEpochsChange,
    cellMetricByCell,
    selectedCellMetric = "cluster_count",
    renderRequestId = 0,
    onRenderComplete,
}: SankeyViewProps) {
    const chartRef = useRef<HTMLDivElement | null>(null);
    const chartPanelRef = useRef<HTMLDivElement | null>(null);
    const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
    const [showSankeyPaths, setShowSankeyPaths] = useState(true);
    const [showSankeyNodes, setShowSankeyNodes] = useState(true);
    const [showCellBackground, setShowCellBackground] = useState(true);
    const [showGapHeatmap, setShowGapHeatmap] = useState(true);
    const [epochTransitionThreshold, setEpochTransitionThreshold] = useState(0);
    const [epochTransitionThresholdUi, setEpochTransitionThresholdUi] = useState(0);
    const maxEpochTransitionPct = EPOCH_THRESHOLD_MAX;
    const [visibleEpochSummary, setVisibleEpochSummary] = useState({ visible: 0, total: 0 });
    const [cellLegendRange, setCellLegendRange] = useState({ min: 0, max: 1 });
    const [gapLegendMax, setGapLegendMax] = useState(1);
    const [linkValueMin, setLinkValueMin] = useState(0);
    const [linkValueMax, setLinkValueMax] = useState(Number.MAX_SAFE_INTEGER);
    const [linkValueMaxBound, setLinkValueMaxBound] = useState(152);
    const autoInitLinkMaxRef = useRef(true);
    const [hoveredCellInfo, setHoveredCellInfo] = useState<CellInfoTag | null>(null);
    const [hoverTooltip, setHoverTooltip] = useState<HoverTooltip | null>(null);
    const [showTooltips, setShowTooltips] = useState(true);
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
        const p = new URLSearchParams(window.location.search);
        let shouldReplace = false;
        if (!p.has("min")) {
            p.set("min", "0");
            shouldReplace = true;
        }
        if (!p.has("max")) {
            p.set("max", "1015");
            shouldReplace = true;
            autoInitLinkMaxRef.current = false;
        }
        if (!p.has("epochMin")) {
            p.set("epochMin", "0");
            shouldReplace = true;
        }
        if (shouldReplace) {
            const next = window.location.pathname + "?" + p.toString();
            window.history.replaceState(null, "", next);
        }

        const q = Number(p.get("epochMin"));
        if (Number.isFinite(q) && q >= 0) {
            const clamped = Math.max(0, Math.min(EPOCH_THRESHOLD_MAX, q));
            setEpochTransitionThreshold(clamped);
            setEpochTransitionThresholdUi(clamped);
        }
        const qMin = Number(p.get("min"));
        const qMax = Number(p.get("max"));
        if (Number.isFinite(qMax) && qMax >= 0) autoInitLinkMaxRef.current = false;
        if (Number.isFinite(qMin) && qMin >= 0) setLinkValueMin(qMin);
        if (Number.isFinite(qMax) && qMax >= 0) setLinkValueMax(qMax);
    }, []);

    useEffect(() => {
        setEpochTransitionThresholdUi(epochTransitionThreshold);
    }, [epochTransitionThreshold]);

    // Recompute link-range upper bound whenever dataset context changes.
    useEffect(() => {
        const renderData = buildRenderData(data, selectedSplits);
        if (!renderData) return;

        let maxLinkValue = 0;
        const connectivity = Array.isArray(renderData.connectivity) ? renderData.connectivity : [];
        connectivity.forEach((targets) => {
            if (!Array.isArray(targets)) return;
            targets.forEach((edge) => {
                const value = Number(Array.isArray(edge) ? edge[1] : 0);
                if (!Number.isFinite(value) || value <= 0) return;
                if (value > maxLinkValue) maxLinkValue = value;
            });
        });

        const nextMaxBound = Math.max(1, Math.ceil(maxLinkValue));
        const p = new URLSearchParams(window.location.search);
        const qMin = Number(p.get("min"));
        const nextMin = Number.isFinite(qMin) && qMin >= 0 ? Math.max(0, Math.min(qMin, nextMaxBound)) : 0;

        setLinkValueMaxBound(nextMaxBound);
        setLinkValueMin(nextMin);
        setLinkValueMax(nextMaxBound);

        const currentMin = p.get("min");
        const currentMax = p.get("max");
        const nextMinStr = String(nextMin);
        const nextMaxStr = String(nextMaxBound);
        if (currentMin !== nextMinStr || currentMax !== nextMaxStr) {
            p.set("min", nextMinStr);
            p.set("max", nextMaxStr);
            window.history.replaceState({}, "", window.location.pathname + "?" + p.toString());
        }
    }, [data, selectedSplits, showFfn2]);

    function applyLinkValueFilter(minVal: number, maxVal: number) {
        const container = chartRef.current;
        if (!container) return;
        const lo = Math.max(0, Math.min(minVal, maxVal));
        const hi = Math.max(lo, maxVal);
        d3.select(container)
            .selectAll<SVGPathElement, unknown>("path.link-path")
            .classed("hidden", function() {
                const v = Number(d3.select(this).attr("data-value") ?? 0) || 0;
                return v < lo || v > hi;
            });
    }

    // this is the actual sankey
    useEffect(() => {
        const container = chartRef.current;
        if (!container || !data) return;
        const renderData = buildRenderData(data, selectedSplits);
        if (!renderData) return;
        let cancelled = false;

        const urlParams = new URLSearchParams(window.location.search);
        let SWAP_AXES = urlParams.get("swap") === "1";

        let svg = d3.select(container).select<SVGSVGElement>("svg.app-sankey");
        if (svg.empty()) {
            svg = d3
                .select(container)
                .append("svg")
                .attr("class", "app-sankey")
                .style("width", "100%")
                .style("height", "100%")
                .attr("preserveAspectRatio", "xMinYMin meet");
        }

        const ro = new ResizeObserver(() => draw());
        ro.observe(container);

        function draw() {
            if (!container) return;
            // Full redraw: remove stale layers/paths from previous renders.
            svg.selectAll("*").remove();
            
            /*
            const isDimenetCase = /\/dimenet\//.test(String(dataPath || "")) || /dataset=dimenet/.test(String(dataPath || ""));
            const shouldHideFfn2Layer = !showFfn2 && !isDimenetCase;
    */

            if (cancelled || chartRef.current !== container) return;

            /*const secondaryNodeValueByCellCluster =
                mergedForCombined?.secondaryNodeValueByCellCluster ?? new Map<string, number>();
    
            const transitionEntriesByEpoch: Array<{
                epoch: string;
                from_layer: string;
                to_layer: string;
                pct_changed: number;
            }> = (
                Array.isArray((transitionJson as any)?.by_epoch) ? (transitionJson as any).by_epoch : []
            ).filter((d: any) => !shouldHideFfn2Layer || (!isHiddenLayerName(String(d?.from_layer ?? "")) && !isHiddenLayerName(String(d?.to_layer ?? ""))));
            const transitionEntriesByLayer: Array<{
                layer: string;
                from_epoch: string;
                to_epoch: string;
                pct_changed: number;
            }> =
                (
                    (Array.isArray((transitionJson as any)?.by_layer) ? (transitionJson as any).by_layer : null) ||
                    (Array.isArray((transitionLayerJson as any)?.by_layer) ? (transitionLayerJson as any).by_layer : [])
                ).filter((d: any) => !shouldHideFfn2Layer || !isHiddenLayerName(String(d?.layer ?? "")));*/

            const allEpochKeys: string[] = sortRowKeys(renderData.epochs);
            const lastEpoch = allEpochKeys.length ? allEpochKeys[allEpochKeys.length - 1] : null;
            const epochPairMaxTransitionPct = new Map<string, number>();
            const epochPairChangedTotal = new Map<string, { changed: number; total: number }>();
            renderData.connectivity.forEach((targets, sourceNodeId) => {
                const srcNode = renderData.nodes[sourceNodeId];
                if (!srcNode || !Array.isArray(targets)) return;
                const sourcePairAgg = new Map<string, { total: number; maxTo: number; byTarget: Map<number, number> }>();
                targets.forEach((edge) => {
                    const targetNodeId = Number(Array.isArray(edge) ? edge[0] : Number.NaN);
                    const count = Number(Array.isArray(edge) ? edge[1] : 0);
                    if (!Number.isFinite(targetNodeId) || !Number.isFinite(count) || count <= 0) return;
                    const tgtNode = renderData.nodes[targetNodeId];
                    if (!tgtNode) return;
                    const srcEpochIdx = Number(srcNode.epoch_idx);
                    const tgtEpochIdx = Number(tgtNode.epoch_idx);
                    if (!Number.isFinite(srcEpochIdx) || !Number.isFinite(tgtEpochIdx)) return;
                    if (Math.abs(srcEpochIdx - tgtEpochIdx) !== 1) return;
                    if (Number(srcNode.layer_idx) !== Number(tgtNode.layer_idx)) return;
                    const fromEpoch = String(srcNode.epoch ?? "").trim();
                    const toEpoch = String(tgtNode.epoch ?? "").trim();
                    if (!fromEpoch || !toEpoch) return;
                    const key = `${fromEpoch}|${toEpoch}`;
                    const pairAgg =
                        sourcePairAgg.get(key) ??
                        { total: 0, maxTo: 0, byTarget: new Map<number, number>() };
                    pairAgg.total += count;
                    const tgtKey = Number(tgtNode.node_id);
                    const nextTo = (pairAgg.byTarget.get(tgtKey) ?? 0) + count;
                    pairAgg.byTarget.set(tgtKey, nextTo);
                    if (nextTo > pairAgg.maxTo) pairAgg.maxTo = nextTo;
                    sourcePairAgg.set(key, pairAgg);
                });
                sourcePairAgg.forEach((pairAgg, key) => {
                    if (pairAgg.total <= 0) return;
                    const changedForSource = Math.max(0, pairAgg.total - pairAgg.maxTo);
                    const agg = epochPairChangedTotal.get(key) ?? { changed: 0, total: 0 };
                    agg.total += pairAgg.total;
                    agg.changed += changedForSource;
                    epochPairChangedTotal.set(key, agg);
                });
            });
            epochPairChangedTotal.forEach((agg, key) => {
                if (agg.total <= 0) return;
                const pct = (agg.changed / agg.total) * 100;
                const prev = epochPairMaxTransitionPct.get(key) ?? 0;
                if (pct > prev) epochPairMaxTransitionPct.set(key, pct);
            });
            const epochOrderIndex = new Map(allEpochKeys.map((e, i) => [e, i]));
            const getEpochPairTransitionPct = (fromEpoch: string, toEpoch: string): number => {
                let best = 0;
                const direct = epochPairMaxTransitionPct.get(`${fromEpoch}|${toEpoch}`) ?? 0;
                const reverse = epochPairMaxTransitionPct.get(`${toEpoch}|${fromEpoch}`) ?? 0;
                best = Math.max(best, direct, reverse);
                const fromIdx = epochOrderIndex.get(fromEpoch);
                const toIdx = epochOrderIndex.get(toEpoch);
                if (fromIdx == null || toIdx == null || fromIdx === toIdx) return best;
                const start = Math.min(fromIdx, toIdx);
                const end = Math.max(fromIdx, toIdx);
                for (let i = start; i < end; i += 1) {
                    const a = allEpochKeys[i];
                    const b = allEpochKeys[i + 1];
                    const seg = Math.max(
                        epochPairMaxTransitionPct.get(`${a}|${b}`) ?? 0,
                        epochPairMaxTransitionPct.get(`${b}|${a}`) ?? 0
                    );
                    if (seg > best) best = seg;
                }
                return best;
            };
            const visibleEpochSet = new Set<string>();
            for (let i = 0; i < allEpochKeys.length - 1; i += 1) {
                const fromEpoch = allEpochKeys[i];
                const toEpoch = allEpochKeys[i + 1];
                const pairPct = getEpochPairTransitionPct(fromEpoch, toEpoch);
                if (pairPct >= epochTransitionThreshold) {
                    visibleEpochSet.add(fromEpoch);
                    visibleEpochSet.add(toEpoch);
                }
            }
            if (lastEpoch != null) visibleEpochSet.add(lastEpoch);
            const visibleEpochBase: string[] = allEpochKeys.filter((epoch) => visibleEpochSet.has(epoch));
            const ROW_NAMES: string[] = reverseEpochOrder ? [...visibleEpochBase].reverse() : visibleEpochBase;
            setVisibleEpochSummary({ visible: ROW_NAMES.length, total: allEpochKeys.length });
            onVisibleEpochsChange?.(visibleEpochBase);
            const COLUMN_NAMES: string[] = sortLayerKeys(renderData.layers);


            const ROW_INDEX = new Map(ROW_NAMES.map((d, i) => [d, i]));
            const COLUMN_INDEX = new Map(COLUMN_NAMES.map((d, i) => [d, i]));

            const X_NAMES = SWAP_AXES ? ROW_NAMES : COLUMN_NAMES;
            const Y_NAMES = SWAP_AXES ? COLUMN_NAMES : ROW_NAMES;
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

            svg.attr("width", width)
                .attr("height", height)
                .attr("viewBox", `0 0 ${width} ${height}`);

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
                    const epochIdx = renderData.epochs.indexOf(rowName);
                    const layerIdx = renderData.layers.indexOf(columnName);
                    const cellIndex = layerIdx + epochIdx * renderData.layers.length;
                    const children = renderData.cells[cellIndex]?.children ?? [];
                    const nodesForCell = children.map((nodeId) => renderData.nodes[nodeId]).filter(Boolean);
                    const ids = new Set<number>();
                    nodesForCell.forEach((n: any) => {
                        const clusterId = Number(n?.cluster_id);
                        const cluster = Number(n?.cluster);
                        const label = String(n?.label ?? "");
                        const labelMatch = /_C(\d+)$/.exec(label);
                        const parsedFromLabel = labelMatch ? Number(labelMatch[1]) : Number.NaN;
                        const c = Number.isFinite(clusterId)
                            ? clusterId
                            : Number.isFinite(cluster)
                                ? cluster
                                : parsedFromLabel;
                        if (Number.isFinite(c)) ids.add(c);
                    });
                    cellClusterCount.set(`${rowName}|${columnName}`, ids.size || nodesForCell.length || 0);
                });
            });
            const cellMetricValue = new Map<string, number>();
            ROW_NAMES.forEach((rowName) => {
                COLUMN_NAMES.forEach((columnName) => {
                    const key = `${rowName}|${columnName}`;
                    if (selectedCellMetric === "cluster_count") {
                        cellMetricValue.set(key, cellClusterCount.get(key) ?? 0);
                        return;
                    }
                    const metricVal = Number(cellMetricByCell?.[key]?.[selectedCellMetric]);
                    if (Number.isFinite(metricVal)) {
                        cellMetricValue.set(key, metricVal);
                    } else {
                        cellMetricValue.set(key, 0);
                    }
                });
            });
            const metricVals = Array.from(cellMetricValue.values());
            const minCellCluster = metricVals.length ? Math.min(...metricVals) : 0;
            const maxCellCluster = metricVals.length ? Math.max(...metricVals) : 1;
            setCellLegendRange((prev) =>
                prev.min === minCellCluster && prev.max === maxCellCluster
                    ? prev
                    : { min: minCellCluster, max: maxCellCluster }
            );
            const cellBgColor = d3
                .scaleLinear<string>()
                .domain(
                    minCellCluster === maxCellCluster
                        ? [minCellCluster, minCellCluster + 1]
                        : [minCellCluster, maxCellCluster]
                )
                .range(["#eff6ff", "#7B8FC7"])
                .clamp(true);

            const cells: d3.Selection<SVGGElement, any, any, any>[] = [];
            let lockedCell: { rowName: string; columnName: string } | null = null;
            const metricNameMap: Record<string, string> = {
                cluster_count: "Cluster Number",
                ari: "ARI",
                ami: "AMI",
                nmi: "NMI",
                homogeneity: "Homogeneity",
                completeness: "Completeness",
                v_measure: "V-Measure",
                fowlkes_mallows: "FMI",
            };
            const activeMetricLabel = metricNameMap[selectedCellMetric] ?? selectedCellMetric;
            const placeTooltipFromEvent = (evt: any, title: string, lines: string[]) => {
                if (!showTooltips) return;
                const panel = chartPanelRef.current;
                const target = evt?.currentTarget as Element | null;
                if (!panel || !target || typeof (target as any).getBoundingClientRect !== "function") return;
                const panelRect = panel.getBoundingClientRect();
                const targetRect = (target as any).getBoundingClientRect() as DOMRect;
                const x = targetRect.left - panelRect.left + targetRect.width / 2;
                const y = targetRect.bottom - panelRect.top;
                setHoverTooltip({ x, y, title, lines });
            };


            const maxClusters = Math.max(1, ...renderData.cells.map((x) => (Array.isArray(x?.children) ? x.children.length : 0)));
            const xScale = d3.scaleBand<number>().range([0, innerW]).domain(d3.range(maxClusters)).padding(0.2);

            const nodeBarWidth = Math.min(xScale.bandwidth(), MAX_NODE_BAR_WIDTH);
            const nodeBarOffset = (xScale.bandwidth() - nodeBarWidth) / 2;
            const cellTotalByIndex = new Map<number, number>();
            renderData.cells.forEach((cell, cellIndex) => {
                const children = Array.isArray(cell?.children) ? cell.children : [];
                const total = children.reduce((sum: number, nodeId: number) => {
                    const node = renderData.nodes[nodeId];
                    return sum + Math.max(0, Number(node?.member_count ?? node?.value ?? 0));
                }, 0);
                cellTotalByIndex.set(cellIndex, total);
            });
            const getNodeBarMetrics = (d: any) => {
                const cellIndex = Number(d?.cell_index ?? -1);
                const cellTotal = Math.max(0, Number(cellTotalByIndex.get(cellIndex) ?? 0));
                const previousValue = Math.max(0, Number(d?.previous ?? 0));
                const memberValue = Math.max(0, Number(d?.member_count ?? d?.value ?? 0));
                if (cellTotal <= 0 || memberValue <= 0) {
                    return { cellTotal, topY: innerH, height: 0 };
                }
                const clampedPrevious = Math.min(cellTotal, previousValue);
                const clampedMember = Math.min(Math.max(0, cellTotal - clampedPrevious), memberValue);
                const topY = (clampedPrevious / cellTotal) * innerH;
                const height = (clampedMember / cellTotal) * innerH;
                return { cellTotal, topY, height };
            };
            const getSplitSegmentHeight = (d: any, splitName: "train" | "test" | "val") => {
                const { cellTotal, height } = getNodeBarMetrics(d);
                if (cellTotal <= 0 || height <= 0) return 0;
                const splitCounts = d?.split_counts ?? {};
                const splitValue = Math.max(0, Number(splitCounts?.[splitName] ?? 0));
                const visibleTotal = Math.max(0, Number(d?.member_count ?? d?.value ?? 0));
                if (visibleTotal <= 0) return 0;
                return (height * Math.min(visibleTotal, splitValue)) / visibleTotal;
            };
            const nodeIndexById = new Map<number, number>();
            renderData.cells.forEach((cell) => {
                const children = Array.isArray(cell?.children) ? cell.children : [];
                children.forEach((nodeId: number, idx: number) => {
                    nodeIndexById.set(nodeId, idx);
                });
            });

            /*const allEdgeValues: number[] = [];
            const maxEdgeVal = d3.max(allEdgeValues) || 1;
            const cellTotals: number[] = nodeDrawQueue.map(({ nodesForCell }) =>
                (Array.isArray(nodesForCell) ? nodesForCell : []).reduce(
                    (sum: number, n: any) => sum + (Number(n?.value ?? 0) || 0),
                    0
                )
            );
            const maxCellTotal = d3.max(cellTotals) || 1;
            const nextMaxBound = Math.max(1, Math.ceil(maxEdgeVal), Math.ceil(maxCellTotal));
            setLinkValueMaxBound(nextMaxBound);
            setLinkValueMin((v) => Math.max(0, Math.min(v, nextMaxBound)));
            if (autoInitLinkMaxRef.current) {
                setLinkValueMax(nextMaxBound);
                const p = new URLSearchParams(window.location.search);
                p.set("max", String(nextMaxBound));
                p.set("min", String(Math.max(0, Math.min(linkValueMin, nextMaxBound))));
                p.set("epochMin", String(epochTransitionThreshold.toFixed(2)));
                window.history.replaceState({}, "", window.location.pathname + "?" + p.toString());
                autoInitLinkMaxRef.current = false;
            } else {
                setLinkValueMax((v) => Math.max(0, Math.min(v, nextMaxBound)));
            }*/
            const linkWidthScale = d3.scaleSqrt().domain([0, Math.max(1, renderData.molecule_ids.length)]).range([0.5, 20]).clamp(true);

            // get horizontal and vertical links
            const intraEpoch: Array<[any, any, number]> = [];
            const intraLayer: Array<[any, any, number]> = [];
            const seenPairs = new Set();
            renderData.connectivity.keys().forEach(function(x) {
                const thisNode = renderData.nodes[x];
                const vals = renderData.connectivity[x];
                const vNodes = vals.map((x) => renderData.nodes[x[0]]);
                vNodes.forEach(function(y, idx) {
                    if (!y) return;
                    const pairKey = `${x}->${y.node_id}`;
                    if (seenPairs.has(pairKey)) return;
                    seenPairs.add(pairKey);
                    // across epochs
                    if (thisNode.layer_idx === y.layer_idx) {
                        intraEpoch.push([thisNode, y, vals[idx][1]])
                    } else {
                        // across layers
                        intraLayer.push([thisNode, y, vals[idx][1]])
                    }
                });
            });
            const getNodesForEpochLayer = (epochName: string, layerName: string): any[] => {
                const epochIdx = renderData.epochs.indexOf(epochName);
                const layerIdx = renderData.layers.indexOf(layerName);
                if (epochIdx < 0 || layerIdx < 0) return [];
                const cellIdx = layerIdx + epochIdx * renderData.layers.length;
                const childIds = renderData.cells[cellIdx]?.children ?? [];
                return childIds.map((nodeId) => renderData.nodes[nodeId]).filter(Boolean);
            };

            const existingEpochLayerPairs = new Set<string>();
            intraEpoch.forEach((entry) => {
                const fromN = entry[0];
                const toN = entry[1];
                const fromEpoch = String(fromN?.epoch ?? "");
                const toEpoch = String(toN?.epoch ?? "");
                const fromLayer = String(fromN?.layer ?? "");
                const toLayer = String(toN?.layer ?? "");
                if (!fromEpoch || !toEpoch || !fromLayer || !toLayer) return;
                if (fromLayer !== toLayer) return;
                existingEpochLayerPairs.add(`${fromEpoch}|${toEpoch}|${fromLayer}`);
                existingEpochLayerPairs.add(`${toEpoch}|${fromEpoch}|${fromLayer}`);
            });
            const syntheticIntraEpoch: Array<[any, any, number]> = [];
            for (let i = 0; i < ROW_NAMES.length - 1; i += 1) {
                const fromEpoch = ROW_NAMES[i];
                const toEpoch = ROW_NAMES[i + 1];
                COLUMN_NAMES.forEach((layerName) => {
                    const pairKey = `${fromEpoch}|${toEpoch}|${layerName}`;
                    if (existingEpochLayerPairs.has(pairKey)) return;
                    const srcNodes = getNodesForEpochLayer(fromEpoch, layerName).filter(
                        (n) => Number(n?.member_count ?? n?.value ?? 0) > 0
                    );
                    const tgtNodes = getNodesForEpochLayer(toEpoch, layerName).filter(
                        (n) => Number(n?.member_count ?? n?.value ?? 0) > 0
                    );
                    if (srcNodes.length === 0 || tgtNodes.length === 0) return;
                    const bridges = buildGreedyBridgeLinks(srcNodes, tgtNodes);
                    if (bridges.length === 0) return;
                    bridges.forEach((bridge) => {
                        const srcNode = srcNodes[bridge.sourceIdx];
                        const tgtNode = tgtNodes[bridge.targetIdx];
                        if (!srcNode || !tgtNode || bridge.value <= 0) return;
                        syntheticIntraEpoch.push([srcNode, tgtNode, bridge.value]);
                    });
                    existingEpochLayerPairs.add(pairKey);
                    existingEpochLayerPairs.add(`${toEpoch}|${fromEpoch}|${layerName}`);
                });
            }

            const drawArcs = (
                dataGroup: Array<[any, any, number]>,
                svgGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
                kind: "column" | "row"
            ) => {
                dataGroup.forEach((entry) => {
                    const fromN = entry[0];
                    const toN = entry[1];
                    const moveCount = Number(entry[2] ?? 0);
                    if (!fromN || !toN || moveCount <= 0) return;
                    const srcIdx = nodeIndexById.get(Number(fromN.node_id)) ?? 0;
                    const tgtIdx = nodeIndexById.get(Number(toN.node_id)) ?? 0;
                    const srcBand = xScale(srcIdx);
                    const tgtBand = xScale(tgtIdx);
                    if (srcBand == null || tgtBand == null) return;

                    const sMetrics = getNodeBarMetrics(fromN);
                    const tMetrics = getNodeBarMetrics(toN);
                    const sBarTop = sMetrics.topY;
                    const sBarH = sMetrics.height;
                    const tBarTop = tMetrics.topY;
                    const tBarH = tMetrics.height;
                    const fromEpochName = String(fromN.epoch ?? "");
                    const fromLayerName = String(fromN.layer ?? "");
                    const toEpochName = String(toN.epoch ?? "");
                    const toLayerName = String(toN.layer ?? "");
                    const srcCellCol = SWAP_AXES ? ROW_INDEX.get(fromEpochName) : COLUMN_INDEX.get(fromLayerName);
                    const srcCellRow = SWAP_AXES ? COLUMN_INDEX.get(fromLayerName) : ROW_INDEX.get(fromEpochName);
                    const tgtCellCol = SWAP_AXES ? ROW_INDEX.get(toEpochName) : COLUMN_INDEX.get(toLayerName);
                    const tgtCellRow = SWAP_AXES ? COLUMN_INDEX.get(toLayerName) : ROW_INDEX.get(toEpochName);
                    if (
                        srcCellCol == null ||
                        srcCellRow == null ||
                        tgtCellCol == null ||
                        tgtCellRow == null
                    ) {
                        return;
                    }
                    const srcCenterX = srcCellCol * cellStepX + cellMargin.left + srcBand + nodeBarOffset + nodeBarWidth / 2;
                    const srcTopY = srcCellRow * cellStepY + cellMargin.top + sBarTop;
                    const srcBottomY = srcTopY + Math.max(0, sBarH);
                    const srcCenterY = (srcTopY + srcBottomY) / 2;
                    const tgtCenterX = tgtCellCol * cellStepX + cellMargin.left + tgtBand + nodeBarOffset + nodeBarWidth / 2;
                    const tgtTopY = tgtCellRow * cellStepY + cellMargin.top + tBarTop;
                    const tgtBottomY = tgtTopY + Math.max(0, tBarH);
                    const tgtCenterY = (tgtTopY + tgtBottomY) / 2;

                    const dxAbs = Math.abs(tgtCenterX - srcCenterX);
                    const dyAbs = Math.abs(tgtCenterY - srcCenterY);
                    const useVerticalAnchors = kind === "column" && dyAbs > 0.5;
                    const upperNode =
                        srcCenterY <= tgtCenterY
                            ? { x: srcCenterX, topY: srcTopY, bottomY: srcBottomY, centerY: srcCenterY }
                            : { x: tgtCenterX, topY: tgtTopY, bottomY: tgtBottomY, centerY: tgtCenterY };
                    const lowerNode =
                        srcCenterY <= tgtCenterY
                            ? { x: tgtCenterX, topY: tgtTopY, bottomY: tgtBottomY, centerY: tgtCenterY }
                            : { x: srcCenterX, topY: srcTopY, bottomY: srcBottomY, centerY: srcCenterY };
                    const startY =
                        useVerticalAnchors
                            ? upperNode.bottomY
                            : (kind === "column" ? srcBottomY : srcCenterY);
                    const endY =
                        useVerticalAnchors
                            ? lowerNode.topY
                            : (kind === "column" ? tgtTopY : tgtCenterY);
                    const startX = useVerticalAnchors ? upperNode.x : srcCenterX;
                    const endX = useVerticalAnchors ? lowerNode.x : tgtCenterX;

                    let pathD = "";
                    const drawDxAbs = Math.abs(endX - startX);
                    const drawDyAbs = Math.abs(endY - startY);
                    const drawVerticalForward = endY >= startY;
                    if (drawDxAbs >= drawDyAbs) {
                        const dx = Math.abs(endX - startX) * 0.45;
                        pathD = `M ${startX},${startY}
                         C ${startX + dx},${startY}
                           ${endX - dx},${endY}
                           ${endX},${endY}`;
                    } else {
                        const dy = Math.abs(endY - startY) * 0.45;
                        pathD = `M ${startX},${startY}
                         C ${startX},${startY + (drawVerticalForward ? dy : -dy)}
                           ${endX},${endY - (drawVerticalForward ? dy : -dy)}
                           ${endX},${endY}`;
                    }

                    const strokeColor = kind === "column" ? SANKEY_2D_COLUMN_LINK_COLOR : SANKEY_2D_ROW_LINK_COLOR;
                    const strokeW = linkWidthScale(moveCount);
                    const path = svgGroup
                        .append("path")
                        .attr("d", pathD)
                        .attr("fill", "none")
                        .attr("stroke", strokeColor)
                        .attr("stroke-width", strokeW)
                        .attr("stroke-opacity", 0.35)
                        .attr("class", "link-path")
                        .attr("data-value", moveCount);

                    if (kind === "column") {
                        path.attr("data-column", String(fromN.layer));
                    } else {
                        path.attr("data-row", String(fromN.epoch));
                    }
                });
            };

            const bgLayer = gRoot.append("g").attr("class", "cell-backgrounds");
            const linkGroup = gRoot.append("g").attr("class", "links column-links");
            const linkGroup2 = gRoot.append("g").attr("class", "links row-links");
            const nodeLayer = gRoot.append("g").attr("class", "cell-nodes");
            const toggleCellLock = (epochName: string, layerName: string) => {
                const isSameLocked =
                    lockedCell?.rowName === epochName &&
                    lockedCell?.columnName === layerName;

                if (isSameLocked) {
                    lockedCell = null;
                    setHoverTooltip(null);
                    onCellSelect?.(false);
                    return;
                }

                lockedCell = { rowName: epochName, columnName: layerName };
                onModelSelect?.(epochName);
                onLayerSelect?.(layerName);
                onCellSelect?.(true);
            };

            drawArcs([...intraEpoch, ...syntheticIntraEpoch], linkGroup, "column");
            drawArcs(intraLayer, linkGroup2, "row");
            // Keep links visually above cells/bars while not blocking interactions on nodes.
            linkGroup.style("pointer-events", "none");
            linkGroup2.style("pointer-events", "none");

            X_NAMES.forEach((xKey, col) => {
                Y_NAMES.forEach((yKey, row) => {
                    const epochName = SWAP_AXES ? xKey : yKey;
                    const layerName = SWAP_AXES ? yKey : xKey;
                    const epochIdx = renderData.epochs.indexOf(epochName);
                    const layerIdx = renderData.layers.indexOf(layerName);
                    if (epochIdx < 0 || layerIdx < 0) return;
                    const cellIdx = layerIdx + epochIdx * renderData.layers.length;

                    bgLayer
                        .append("rect")
                        .attr("id", `cell-bg-${epochName}-${layerName}`)
                        .attr("class", "cell-bg")
                        .attr("x", col * cellStepX)
                        .attr("y", row * cellStepY)
                        .attr("width", cellW)
                        .attr("height", cellH)
                        .attr("fill", () => {
                            if (!showCellBackground) return "white";
                            const key = `${epochName}|${layerName}`;
                            const metricValue = Number(cellMetricValue.get(key) ?? 0);
                            return cellBgColor(metricValue);
                        })
                        .attr("stroke", "#888")
                        .attr("data-row", epochName)
                        .attr("data-column", layerName)
                        .attr("data-cluster-count", renderData.cells[cellIdx]?.children?.length ?? 0)
                        .style("cursor", "pointer")
                        .on("mouseenter", (event: any) => {
                            setHoveredCellInfo({
                                rowName: epochName,
                                columnName: layerName,
                                epochName,
                                layerName,
                                col,
                                row,
                            });
                            const key = `${epochName}|${layerName}`;
                            const clusterCount = Number(cellClusterCount.get(key) ?? 0);
                            const metricValue = Number(cellMetricValue.get(key) ?? 0);
                            const lines = [
                                `Layer: ${layerName}`,
                                `Epoch: ${epochName}`,
                                `Cluster Number: ${clusterCount.toFixed(0)}`,
                            ];
                            if (selectedCellMetric !== "cluster_count") {
                                lines.push(`${activeMetricLabel}: ${Number.isFinite(metricValue) ? metricValue.toFixed(3) : "0.000"}`);
                            }
                            placeTooltipFromEvent(event, "Cell Metrics", lines);
                        })
                        .on("mousemove", (event: any) => {
                            const key = `${epochName}|${layerName}`;
                            const clusterCount = Number(cellClusterCount.get(key) ?? 0);
                            const metricValue = Number(cellMetricValue.get(key) ?? 0);
                            const lines = [
                                `Layer: ${layerName}`,
                                `Epoch: ${epochName}`,
                                `Cluster Number: ${clusterCount.toFixed(0)}`,
                            ];
                            if (selectedCellMetric !== "cluster_count") {
                                lines.push(`${activeMetricLabel}: ${Number.isFinite(metricValue) ? metricValue.toFixed(3) : "0.000"}`);
                            }
                            placeTooltipFromEvent(event, "Cell Metrics", lines);
                        })
                        .on("mouseleave", () => {
                            setHoveredCellInfo(null);
                            setHoverTooltip(null);
                        })
                        .on("click", (event: any) => {
                            event.stopPropagation();
                            toggleCellLock(epochName, layerName);
                        });

                    const g = nodeLayer
                        .append("g")
                        .attr("id", `cell-${epochName}-${layerName}`)
                        .attr("transform", `translate(${col * cellStepX}, ${row * cellStepY})`)
                        .datum({ rowName: epochName, columnName: layerName, col, row });

                    const innerG = g.append("g").attr("transform", `translate(${cellMargin.left}, ${cellMargin.top})`);
                    cells.push(innerG);

                    const nodesForCell = (renderData.cells[cellIdx]?.children ?? []).map((x) => renderData.nodes[x]);
                    const bars = innerG.selectAll("rect.bar-main")
                        .data(nodesForCell)
                        .enter()
                        .append("rect")
                        .attr("class", "bar bar-main")
                        .attr("x", (_d: any, i: number) => (xScale(i) as number) + nodeBarOffset)
                        .attr("y", (d: any) => getNodeBarMetrics(d).topY)
                        .attr("width", nodeBarWidth)
                        .attr("height", (d: any) => getNodeBarMetrics(d).height)
                        .attr("fill", "#aaa")
                        .attr("data-row", epochName)
                        .attr("data-column", layerName)
                        .attr("data-cluster", (d: any) => {
                            const c = d?.cluster_id;
                            if (c !== null && c !== undefined && c !== "") return Number(c);
                            const label = String(d?.label ?? "");
                            const i = label.lastIndexOf("_C");
                            if (i === -1) return "";
                            return Number(label.slice(i + 2));
                        });

                    innerG.selectAll("rect.bar-test")
                        .data(nodesForCell)
                        .enter()
                        .append("rect")
                        .attr("class", "bar bar-test")
                        .attr("x", (_d: any, i: number) => (xScale(i) as number) + nodeBarOffset)
                        .attr("y", (d: any) => {
                            const { topY, height } = getNodeBarMetrics(d);
                            const valHeight = getSplitSegmentHeight(d, "val");
                            const testHeight = getSplitSegmentHeight(d, "test");
                            return topY + Math.max(0, height - valHeight - testHeight);
                        })
                        .attr("width", nodeBarWidth)
                        .attr("height", (d: any) => getSplitSegmentHeight(d, "test"))
                        .attr("fill", "#888")
                        .attr("data-row", epochName)
                        .attr("data-column", layerName)
                        .attr("data-cluster", (d: any) => {
                            const c = d?.cluster_id;
                            if (c !== null && c !== undefined && c !== "") return Number(c);
                            const label = String(d?.label ?? "");
                            const i = label.lastIndexOf("_C");
                            if (i === -1) return "";
                            return Number(label.slice(i + 2));
                        })
                        .style("pointer-events", "none");

                    innerG.selectAll("rect.bar-val")
                        .data(nodesForCell)
                        .enter()
                        .append("rect")
                        .attr("class", "bar bar-val")
                        .attr("x", (_d: any, i: number) => (xScale(i) as number) + nodeBarOffset)
                        .attr("y", (d: any) => {
                            const { topY, height } = getNodeBarMetrics(d);
                            const valHeight = getSplitSegmentHeight(d, "val");
                            return topY + Math.max(0, height - valHeight);
                        })
                        .attr("width", nodeBarWidth)
                        .attr("height", (d: any) => getSplitSegmentHeight(d, "val"))
                        .attr("fill", "#666")
                        .attr("data-row", epochName)
                        .attr("data-column", layerName)
                        .attr("data-cluster", (d: any) => {
                            const c = d?.cluster_id;
                            if (c !== null && c !== undefined && c !== "") return Number(c);
                            const label = String(d?.label ?? "");
                            const i = label.lastIndexOf("_C");
                            if (i === -1) return "";
                            return Number(label.slice(i + 2));
                        })
                        .style("pointer-events", "none");

                    bars
                        .style("cursor", "pointer")
                        .on("click", (event: any, d: any) => {
                            event.stopPropagation();
                            const clusterFromAttr = Number(
                                d3.select(event.currentTarget as SVGRectElement).attr("data-cluster")
                            );
                            const clusterFromData =
                                d?.cluster_id != null && d?.cluster_id !== ""
                                    ? Number(d.cluster_id)
                                    : parseClusterFromLabel(String(d?.label ?? ""));
                            const c = Number.isFinite(clusterFromAttr)
                                ? clusterFromAttr
                                : Number.isFinite(clusterFromData)
                                    ? clusterFromData
                                    : Number.NaN;
                            if (!Number.isFinite(c)) {
                                toggleCellLock(epochName, layerName);
                                return;
                            }
                            onNodeSelect?.({ model: epochName, layer: layerName, cluster: c });
                        });
                })

                /* g.on("click", (_event: any, datum: any) => {
                     const isSameLocked =
                         lockedCell?.rowName === datum.rowName &&
                         lockedCell?.columnName === datum.columnName;
     
                     // Toggle off when clicking the same locked cell.
                     if (isSameLocked) {
                         lockedCell = null;
                         setPinnedCellInfo(null);
                         setHoverTooltip(null);
                         //setCellHighlight(datum.rowName, datum.columnName, false);
                         onCellSelect?.(false);
                         return;
                     }
     
                     lockedCell = { rowName: datum.rowName, columnName: datum.columnName };
                     const epochName = MODEL_SET.has(datum.rowName)
                         ? datum.rowName
                         : MODEL_SET.has(datum.columnName)
                             ? datum.columnName
                             : datum.rowName;
                     const layerName = LAYER_SET.has(datum.rowName)
                         ? datum.rowName
                         : LAYER_SET.has(datum.columnName)
                             ? datum.columnName
                             : datum.columnName;
                     setPinnedCellInfo({
                         rowName: datum.rowName,
                         columnName: datum.columnName,
                         epochName,
                         layerName,
                         col: Number(datum.col ?? 0),
                         row: Number(datum.row ?? 0),
                     });
                     //setCellHighlight(datum.rowName, datum.columnName, true);
     
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
                 })
                     .on("mouseenter", (event: any, datum: any) => {
                         const epochName = MODEL_SET.has(datum.rowName)
                             ? datum.rowName
                             : MODEL_SET.has(datum.columnName)
                                 ? datum.columnName
                                 : datum.rowName;
                         const layerName = LAYER_SET.has(datum.rowName)
                             ? datum.rowName
                             : LAYER_SET.has(datum.columnName)
                                 ? datum.columnName
                                 : datum.columnName;
                         setHoveredCellInfo({
                             rowName: datum.rowName,
                             columnName: datum.columnName,
                             epochName,
                             layerName,
                             col: Number(datum.col ?? 0),
                             row: Number(datum.row ?? 0),
                         });
                         const key = `${datum.rowName}|${datum.columnName}`;
                         const clusterCount = Number(cellClusterCount.get(key) ?? 0);
                         const metricValue = Number(cellMetricValue.get(key) ?? 0);
                         const lines = [
                             `Layer: ${layerName}`,
                             `Epoch: ${epochName}`,
                             `Cluster Number: ${clusterCount.toFixed(0)}`,
                         ];
                         if (selectedCellMetric !== "cluster_count") {
                             lines.push(`${activeMetricLabel}: ${Number.isFinite(metricValue) ? metricValue.toFixed(3) : "0.000"}`);
                         }
                         placeTooltipFromEvent(event, "Cell Metrics", lines);
                     })
                     .on("mousemove", (_event: any, datum: any) => {
                         const epochName = MODEL_SET.has(datum.rowName)
                             ? datum.rowName
                             : MODEL_SET.has(datum.columnName)
                                 ? datum.columnName
                                 : datum.rowName;
                         const layerName = LAYER_SET.has(datum.rowName)
                             ? datum.rowName
                             : LAYER_SET.has(datum.columnName)
                                 ? datum.columnName
                                 : datum.columnName;
                         const key = `${datum.rowName}|${datum.columnName}`;
                         const clusterCount = Number(cellClusterCount.get(key) ?? 0);
                         const metricValue = Number(cellMetricValue.get(key) ?? 0);
                         const lines = [
                             `Layer: ${layerName}`,
                             `Epoch: ${epochName}`,
                             `Cluster Number: ${clusterCount.toFixed(0)}`,
                         ];
                         if (selectedCellMetric !== "cluster_count") {
                             lines.push(`${activeMetricLabel}: ${Number.isFinite(metricValue) ? metricValue.toFixed(3) : "0.000"}`);
                         }
                         placeTooltipFromEvent(_event, "Cell Metrics", lines);
                     })
                     .on("mouseleave", () => {
                         setHoveredCellInfo(null);
                         setHoverTooltip(null);
                     });
             });*/
            });

            // Ensure links stay on top after all cells/bars are appended.
            linkGroup.raise();
            linkGroup2.raise();



            type GapSourceAgg = {
                total: number;
                maxTo: number;
                byTarget: Map<number, number>;
            };
            const layerGapAggByPair = new Map<string, Map<number, GapSourceAgg>>();
            const epochGapAggByPair = new Map<string, Map<number, GapSourceAgg>>();
            const addGapEdge = (
                byPair: Map<string, Map<number, GapSourceAgg>>,
                pairKey: string,
                sourceNodeId: number,
                targetNodeId: number,
                value: number
            ) => {
                if (!Number.isFinite(sourceNodeId) || !Number.isFinite(targetNodeId)) return;
                if (!Number.isFinite(value) || value <= 0) return;
                const bySource = byPair.get(pairKey) ?? new Map<number, GapSourceAgg>();
                const sourceAgg = bySource.get(sourceNodeId) ?? {
                    total: 0,
                    maxTo: 0,
                    byTarget: new Map<number, number>(),
                };
                sourceAgg.total += value;
                const nextTo = (sourceAgg.byTarget.get(targetNodeId) ?? 0) + value;
                sourceAgg.byTarget.set(targetNodeId, nextTo);
                if (nextTo > sourceAgg.maxTo) sourceAgg.maxTo = nextTo;
                bySource.set(sourceNodeId, sourceAgg);
                byPair.set(pairKey, bySource);
            };
            const pctFromGapAgg = (bySource: Map<number, GapSourceAgg> | undefined): number => {
                if (!bySource || bySource.size === 0) return 0;
                let total = 0;
                let changed = 0;
                bySource.forEach((sourceAgg) => {
                    if (!Number.isFinite(sourceAgg.total) || sourceAgg.total <= 0) return;
                    total += sourceAgg.total;
                    changed += Math.max(0, sourceAgg.total - sourceAgg.maxTo);
                });
                if (total <= 0) return 0;
                return (changed / total) * 100;
            };

            renderData.connectivity.forEach((targets, sourceNodeId) => {
                const srcNode = renderData.nodes[sourceNodeId];
                if (!srcNode || !Array.isArray(targets)) return;
                const srcEpoch = String(srcNode.epoch ?? "");
                const srcLayer = String(srcNode.layer ?? "");
                const srcRowIdx = ROW_INDEX.get(srcEpoch);
                const srcColIdx = COLUMN_INDEX.get(srcLayer);
                if (srcRowIdx == null || srcColIdx == null) return;

                targets.forEach((edge) => {
                    const targetNodeId = Number(Array.isArray(edge) ? edge[0] : Number.NaN);
                    const value = Number(Array.isArray(edge) ? edge[1] : 0);
                    if (!Number.isFinite(targetNodeId) || !Number.isFinite(value) || value <= 0) return;
                    const tgtNode = renderData.nodes[targetNodeId];
                    if (!tgtNode) return;
                    const tgtEpoch = String(tgtNode.epoch ?? "");
                    const tgtLayer = String(tgtNode.layer ?? "");
                    const tgtRowIdx = ROW_INDEX.get(tgtEpoch);
                    const tgtColIdx = COLUMN_INDEX.get(tgtLayer);
                    if (tgtRowIdx == null || tgtColIdx == null) return;

                    if (srcEpoch === tgtEpoch && Math.abs(srcColIdx - tgtColIdx) === 1) {
                        const forward = srcColIdx <= tgtColIdx;
                        const fromColIdx = forward ? srcColIdx : tgtColIdx;
                        const toColIdx = forward ? tgtColIdx : srcColIdx;
                        const fromLayer = COLUMN_NAMES[fromColIdx];
                        const toLayer = COLUMN_NAMES[toColIdx];
                        const pairKey = `${srcEpoch}|${fromLayer}|${toLayer}`;
                        const fromNodeId = forward ? Number(srcNode.node_id) : Number(tgtNode.node_id);
                        const toNodeId = forward ? Number(tgtNode.node_id) : Number(srcNode.node_id);
                        addGapEdge(layerGapAggByPair, pairKey, fromNodeId, toNodeId, value);
                    }

                    if (srcLayer === tgtLayer && Math.abs(srcRowIdx - tgtRowIdx) === 1) {
                        const forward = srcRowIdx <= tgtRowIdx;
                        const fromRowIdx = forward ? srcRowIdx : tgtRowIdx;
                        const toRowIdx = forward ? tgtRowIdx : srcRowIdx;
                        const fromEpoch = ROW_NAMES[fromRowIdx];
                        const toEpoch = ROW_NAMES[toRowIdx];
                        const pairKey = `${srcLayer}|${fromEpoch}|${toEpoch}`;
                        const fromNodeId = forward ? Number(srcNode.node_id) : Number(tgtNode.node_id);
                        const toNodeId = forward ? Number(tgtNode.node_id) : Number(srcNode.node_id);
                        addGapEdge(epochGapAggByPair, pairKey, fromNodeId, toNodeId, value);
                    }
                });
            });

            const getLayerGapPct = (epoch: string, fromLayer: string, toLayer: string): number => {
                const key = `${epoch}|${fromLayer}|${toLayer}`;
                return pctFromGapAgg(layerGapAggByPair.get(key));
            };
            const getEpochGapPct = (layer: string, fromEpoch: string, toEpoch: string): number => {
                const key = `${layer}|${fromEpoch}|${toEpoch}`;
                return pctFromGapAgg(epochGapAggByPair.get(key));
            };
            const allGapPcts: number[] = [];
            ROW_NAMES.forEach((epoch) => {
                for (let i = 0; i < COLUMN_NAMES.length - 1; i += 1) {
                    allGapPcts.push(getLayerGapPct(epoch, COLUMN_NAMES[i], COLUMN_NAMES[i + 1]));
                }
            });
            COLUMN_NAMES.forEach((layer) => {
                for (let i = 0; i < ROW_NAMES.length - 1; i += 1) {
                    allGapPcts.push(getEpochGapPct(layer, ROW_NAMES[i], ROW_NAMES[i + 1]));
                }
            });
            const maxGapPct = Math.max(1, ...allGapPcts.filter((v) => Number.isFinite(v) && v >= 0));
            setGapLegendMax((prev) => (prev === maxGapPct ? prev : maxGapPct));
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
                    ROW_NAMES.forEach((epoch, rowIdx) => {
                        for (let i = 0; i < COLUMN_NAMES.length - 1; i += 1) {
                            const fromLayer = COLUMN_NAMES[i];
                            const toLayer = COLUMN_NAMES[i + 1];
                            const pct = getLayerGapPct(epoch, fromLayer, toLayer);
                            const xGap = i * cellStepX + cellW;
                            const yGap = rowIdx * cellStepY;
                            gapHeatmap
                                .append("rect")
                                .attr("class", "gap-rect")
                                .attr("x", xGap)
                                .attr("y", yGap)
                                .attr("width", CELL_GAP)
                                .attr("height", cellH)
                                .attr("fill", horizontalGapColor(pct))
                                .attr("stroke", "#ffffff")
                                .attr("stroke-width", 0.7)
                                .attr("data-row", epoch)
                                .attr("data-column-from", fromLayer)
                                .attr("data-column-to", toLayer)
                                .on("mouseenter mousemove", (event: any) => {
                                    placeTooltipFromEvent(event, "Gap Transition", [
                                        `${epoch}: ${fromLayer} -> ${toLayer}`,
                                        `${pct.toFixed(2)}% changed to other clusters`,
                                    ]);
                                })
                                .on("mouseleave", () => setHoverTooltip(null));
                        }
                    });
                    COLUMN_NAMES.forEach((layer, colIdx) => {
                        for (let i = 0; i < ROW_NAMES.length - 1; i += 1) {
                            const fromEpoch = ROW_NAMES[i];
                            const toEpoch = ROW_NAMES[i + 1];
                            const pct = getEpochGapPct(layer, fromEpoch, toEpoch);
                            const xGap = colIdx * cellStepX;
                            const yGap = i * cellStepY + cellH;
                            gapHeatmap
                                .append("rect")
                                .attr("class", "gap-rect")
                                .attr("x", xGap)
                                .attr("y", yGap)
                                .attr("width", cellW)
                                .attr("height", CELL_GAP)
                                .attr("fill", verticalGapColor(pct))
                                .attr("stroke", "#ffffff")
                                .attr("stroke-width", 0.7)
                                .attr("data-column", layer)
                                .attr("data-row-from", fromEpoch)
                                .attr("data-row-to", toEpoch)
                                .on("mouseenter mousemove", (event: any) => {
                                    placeTooltipFromEvent(event, "Gap Transition", [
                                        `${layer}: ${fromEpoch} -> ${toEpoch}`,
                                        `${pct.toFixed(2)}% changed to other clusters`,
                                    ]);
                                })
                                .on("mouseleave", () => setHoverTooltip(null));
                        }
                    });
                } else {
                    COLUMN_NAMES.forEach((layer, rowIdx) => {
                        for (let i = 0; i < ROW_NAMES.length - 1; i += 1) {
                            const fromEpoch = ROW_NAMES[i];
                            const toEpoch = ROW_NAMES[i + 1];
                            const pct = getEpochGapPct(layer, fromEpoch, toEpoch);
                            const xGap = i * cellStepX + cellW;
                            const yGap = rowIdx * cellStepY;
                            gapHeatmap
                                .append("rect")
                                .attr("class", "gap-rect")
                                .attr("x", xGap)
                                .attr("y", yGap)
                                .attr("width", CELL_GAP)
                                .attr("height", cellH)
                                .attr("fill", horizontalGapColor(pct))
                                .attr("stroke", "#ffffff")
                                .attr("stroke-width", 0.7)
                                .attr("data-column", layer)
                                .attr("data-row-from", fromEpoch)
                                .attr("data-row-to", toEpoch)
                                .on("mouseenter mousemove", (event: any) => {
                                    placeTooltipFromEvent(event, "Gap Transition", [
                                        `${layer}: ${fromEpoch} -> ${toEpoch}`,
                                        `${pct.toFixed(2)}% changed to other clusters`,
                                    ]);
                                })
                                .on("mouseleave", () => setHoverTooltip(null));
                        }
                    });
                    ROW_NAMES.forEach((epoch, colIdx) => {
                        for (let i = 0; i < COLUMN_NAMES.length - 1; i += 1) {
                            const fromLayer = COLUMN_NAMES[i];
                            const toLayer = COLUMN_NAMES[i + 1];
                            const pct = getLayerGapPct(epoch, fromLayer, toLayer);
                            const xGap = colIdx * cellStepX;
                            const yGap = i * cellStepY + cellH;
                            gapHeatmap
                                .append("rect")
                                .attr("class", "gap-rect")
                                .attr("x", xGap)
                                .attr("y", yGap)
                                .attr("width", cellW)
                                .attr("height", CELL_GAP)
                                .attr("fill", verticalGapColor(pct))
                                .attr("stroke", "#ffffff")
                                .attr("stroke-width", 0.7)
                                .attr("data-row", epoch)
                                .attr("data-column-from", fromLayer)
                                .attr("data-column-to", toLayer)
                                .on("mouseenter mousemove", (event: any) => {
                                    placeTooltipFromEvent(event, "Gap Transition", [
                                        `${epoch}: ${fromLayer} -> ${toLayer}`,
                                        `${pct.toFixed(2)}% changed to other clusters`,
                                    ]);
                                })
                                .on("mouseleave", () => setHoverTooltip(null));
                        }
                    });
                }
            }

            function buildGreedyBridgeLinks(
                sourceNodes: any[],
                targetNodes: any[]
            ): Array<{ sourceIdx: number; targetIdx: number; value: number }> {
                const clusterOf = (n: any, fallback: number) => {
                    const raw = n?.cluster_id ?? n?.cluster ?? n?.id;
                    const numeric = Number(raw);
                    if (Number.isFinite(numeric)) return numeric;
                    const parsed = parseClusterFromLabel(String(n?.label ?? ""));
                    return Number.isFinite(parsed) ? parsed : fallback;
                };
                const massOf = (n: any) => Math.max(0, Number(n?.member_count ?? n?.value ?? 0));
                const src = sourceNodes
                    .map((n, i) => ({ i, cluster: clusterOf(n, i), rem: massOf(n) }))
                    .filter((d) => d.rem > 0)
                    .sort((a, b) => a.cluster - b.cluster);
                const tgt = targetNodes
                    .map((n, i) => ({ i, cluster: clusterOf(n, i), rem: massOf(n) }))
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


            // Ensure every adjacent visible epoch pair has paths within this layer.
            // If source data lacks a direct visible pair after filtering, synthesize conservative bridges.
            /*for (let i = 0; i < ROW_NAMES.length - 1; i += 1) {
                const fromRow = ROW_NAMES[i];
                const toRow = ROW_NAMES[i + 1];
                const pairKey = `${fromRow}->${toRow}`;
                if (acceptedEpochPairSet.has(pairKey)) continue;

                const srcNodes = ((columnObj?.nodes?.[fromRow] || []) as any[]).filter((n) => (n?.value ?? 0) > 0);
                const tgtNodes = ((columnObj?.nodes?.[toRow] || []) as any[]).filter((n) => (n?.value ?? 0) > 0);
                if (srcNodes.length === 0 || tgtNodes.length === 0) continue;

                const bridges = buildGreedyBridgeLinks(srcNodes, tgtNodes);
                const srcO = cellOrigin(fromRow, columnKey, cellStepX, cellStepY);
                const tgtO = cellOrigin(toRow, columnKey, cellStepX, cellStepY);

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

                    const syntheticW = Math.max(0.8, linkWidthScale(bridge.value));
                    const synthetic = appendLinkPath(
                        linkGroup as any,
                        pathD,
                        bridge.value,
                        syntheticW,
                        SANKEY_2D_COLUMN_LINK_COLOR,
                        0.18
                    );
                    synthetic.attr("data-column", columnKey).attr("data-synthetic", "1");
                });
            }
        });*/

            /*// Ensure every adjacent layer pair has ordered paths.
                // If raw edges have none for a pair, synthesize a complete conservative bridge
                // from both sides' node masses (cluster-wise), not a single coarse link.
                const getNodesForColumn = (columnName: string): any[] => {
                    for (const alias of layerNameAliases(columnName)) {
                        const arr = rowObj?.nodes?.[alias];
                        if (Array.isArray(arr) && arr.length > 0) return arr;
                    }
                    return [];
                };
                for (let i = 0; i < COLUMN_NAMES.length - 1; i += 1) {
                    const fromCol = resolveExistingColumnName(COLUMN_NAMES[i]);
                    const toCol = resolveExistingColumnName(COLUMN_NAMES[i + 1]);
                    const pairKey = `${fromCol}->${toCol}`;
                    if (acceptedPairSet.has(pairKey)) continue;

                    const srcNodes = (getNodesForColumn(fromCol) as any[]).filter((n) => (n?.value ?? 0) > 0);
                    const tgtNodes = (getNodesForColumn(toCol) as any[]).filter((n) => (n?.value ?? 0) > 0);
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
            });*/




            // Enforce final paint order: background (bottom) -> links -> nodes (top).
            bgLayer.lower();
            linkGroup.raise();
            linkGroup2.raise();
            nodeLayer.raise();

            const btnDefault = document.getElementById("btnDefault") as HTMLButtonElement | null;
            const btnColumn = document.getElementById("btnColumn") as HTMLButtonElement | null;
            const btnRow = document.getElementById("btnRow") as HTMLButtonElement | null;
            const btnSwap = document.getElementById("btnSwap") as HTMLButtonElement | null;
            const setSwapParam = (swapOn: boolean) => {
                SWAP_AXES = swapOn;
                const p = new URLSearchParams(window.location.search);
                p.set("swap", swapOn ? "1" : "0");
                window.history.replaceState({}, "", window.location.pathname + "?" + p.toString());
                draw();
                onSwapAxes?.(swapOn);
            };

            function setMode(m: "default" | "column" | "row") {
                if (btnDefault && btnColumn && btnRow) {
                    btnDefault.style.fontWeight = m === "default" ? "600" : "400";
                    btnColumn.style.fontWeight = m === "column" ? "600" : "400";
                    btnRow.style.fontWeight = m === "row" ? "600" : "400";
                }

                if (m === "default") {
                    d3.selectAll('#chart svg rect.cell-bg').classed("cell-dimmed", false).classed("cell-highlight", false);
                    d3.selectAll('#chart svg g[id^="cell-"] rect.bar').classed("cell-dimmed", false);
                    d3.selectAll("path.link-path").classed("link-dimmed", false).classed("link-highlight", false);
                    d3.selectAll("rect.gap-rect").classed("gap-dimmed", false);
                } else if (m === "column") {
                    d3.selectAll("path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
                    d3.selectAll("rect.gap-rect").classed("gap-dimmed", false);
                    if (!SWAP_AXES) {
                        d3.selectAll("g.column-links path.link-path").classed("link-dimmed", false).classed("link-highlight", true);
                        d3.selectAll("g.row-links path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
                    } else {
                        d3.selectAll("g.row-links path.link-path").classed("link-dimmed", false).classed("link-highlight", true);
                        d3.selectAll("g.column-links path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
                    }
                } else if (m === "row") {
                    d3.selectAll("path.link-path").classed("link-dimmed", true).classed("link-highlight", false);
                    d3.selectAll("rect.gap-rect").classed("gap-dimmed", false);
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
            applyLinkValueFilter(linkValueMin, Math.min(linkValueMax, linkValueMaxBound));
            if (!cancelled && chartRef.current === container) {
                onRenderComplete?.(renderRequestId);
            }

        }
        return () => {
            cancelled = true;
            ro.disconnect();
        };
    }, [
        data,
        transitionStatsPath,
        showCellBackground,
        showGapHeatmap,
        showTooltips,
        epochTransitionThreshold,
        reverseEpochOrder,
        onVisibleEpochsChange,
        selectedCellMetric,
        cellMetricByCell,
        showFfn2,
        selectedSplits,
        renderRequestId,
        onRenderComplete,
    ]);

    useEffect(() => {
        applyLinkValueFilter(linkValueMin, Math.min(linkValueMax, linkValueMaxBound));
    }, [linkValueMin, linkValueMax, linkValueMaxBound]);

    useEffect(() => {
        const container = chartRef.current;
        if (!container) return;
        d3.select(container)
            .selectAll<SVGGElement, unknown>("g.links")
            .style("display", showSankeyPaths ? null : "none");
        d3.select(container)
            .selectAll<SVGGElement, unknown>("g.cell-nodes")
            .style("display", showSankeyNodes ? null : "none");
    }, [showSankeyPaths, showSankeyNodes]);

    useEffect(() => {
        const container = chartRef.current;
        if (!container) return;
        const sel = d3.select(container).selectAll<SVGRectElement, any>("rect.bar");
        const darker = (hex: string, amount = 0.6) => {
            const c = d3.color(hex);
            return c ? c.darker(amount).formatHex() : hex;
        };
        const neutralFill = (el: d3.Selection<SVGRectElement, any, any, any>) => {
            if (el.classed("bar-val")) return "#666";
            if (el.classed("bar-test")) return "#888";
            return "#aaa";
        };
        const colorizedFill = (el: d3.Selection<SVGRectElement, any, any, any>, base: string) => {
            if (el.classed("bar-val")) return darker(base, 1.0);
            if (el.classed("bar-test")) return darker(base, 0.6);
            return base;
        };
        const focusModel = selectedNode?.model ?? activeCell?.model ?? null;
        const focusLayer = selectedNode?.layer ?? activeCell?.layer ?? null;
        const inColorCell = (row: string, col: string) => {
            if (selectedNode) return row === selectedNode.model && col === selectedNode.layer;
            return row === focusModel && col === focusLayer;
        };
        if (!focusModel || !focusLayer) {
            sel.attr("fill", function() {
                return neutralFill(d3.select(this));
            });
            sel.classed("node-highlight", false).classed("node-dimmed", false).classed("node-context", false);
            d3.select(container)
                .selectAll<SVGRectElement, any>('svg rect.cell-bg')
                .classed("cell-dimmed", false)
                .classed("cell-highlight", false);
            d3.select(container)
                .selectAll<SVGRectElement, any>('svg g[id^="cell-"] rect.bar')
                .classed("cell-dimmed", false);
            d3.select(container)
                .selectAll("path.link-path")
                .classed("link-dimmed", false)
                .classed("link-highlight", false);
            d3.select(container)
                .selectAll("rect.gap-rect")
                .classed("gap-dimmed", false);
            return;
        }
        const clusterValues: number[] = [];
        sel.each(function() {
            const el = d3.select(this);
            const row = el.attr("data-row");
            const col = el.attr("data-column");
            if (!inColorCell(row, col)) return;
            const cluster = Number(el.attr("data-cluster"));
            if (Number.isFinite(cluster)) clusterValues.push(cluster);
        });
        const maxCluster = clusterValues.length ? Math.max(...clusterValues) : 0;
        const color = d3
            .scaleOrdinal<number, string>()
            .domain(d3.range(maxCluster + 1))
            .range(CLUSTER_COLORS);
        sel.each(function() {
            const el = d3.select(this);
            const row = el.attr("data-row");
            const col = el.attr("data-column");
            const cluster = Number(el.attr("data-cluster"));
            if (inColorCell(row, col) && Number.isFinite(cluster)) {
                const base = color(Math.max(0, cluster));
                el.attr("fill", colorizedFill(el, base));
            } else {
                el.attr("fill", neutralFill(el));
            }
            if (!selectedNode) {
                el.classed("node-highlight", false);
                el.classed("node-dimmed", false);
                el.classed("node-context", false);
                return;
            }
            const isRowOrCol =
                row === selectedNode.model ||
                col === selectedNode.layer;
            const isMatch =
                row === selectedNode.model &&
                col === selectedNode.layer &&
                Number.isFinite(cluster) &&
                cluster === selectedNode.cluster;
            el.classed("node-highlight", isMatch);
            el.classed("node-context", isRowOrCol && !isMatch);
            el.classed("node-dimmed", !isRowOrCol);
        });

        if (!focusModel || !focusLayer) return;

        const rowKey = focusModel;
        const colKey = focusLayer;
        const cellSel = d3.select(container).selectAll('svg rect.cell-bg, svg g[id^="cell-"] rect.bar');
        cellSel.classed("cell-dimmed", true).classed("cell-highlight", false);

        d3.select(container)
            .selectAll<SVGRectElement, any>('svg rect.cell-bg')
            .each(function() {
                const bg = d3.select(this);
                const row = String(bg.attr("data-row") ?? "");
                const col = String(bg.attr("data-column") ?? "");
                if (row === rowKey || col === colKey) {
                    bg.classed("cell-dimmed", false).classed("cell-highlight", true);
                }
            });
        d3.select(container)
            .selectAll<SVGRectElement, any>('svg g[id^="cell-"] rect.bar')
            .each(function() {
                const bar = d3.select(this);
                const row = String(bar.attr("data-row") ?? "");
                const col = String(bar.attr("data-column") ?? "");
                if (row === rowKey || col === colKey) {
                    bar.classed("cell-dimmed", false);
                }
            });

        d3.select(container)
            .selectAll("path.link-path")
            .classed("link-dimmed", true)
            .classed("link-highlight", false);
        d3.select(container)
            .selectAll("rect.gap-rect")
            .classed("gap-dimmed", true);
        d3.select(container)
            .selectAll("g.row-links path.link-path")
            .each(function() {
                const el = d3.select(this);
                if (el.attr("data-row") === rowKey) el.classed("link-dimmed", false).classed("link-highlight", true);
            });
        d3.select(container)
            .selectAll("g.column-links path.link-path")
            .each(function() {
                const el = d3.select(this);
                if (el.attr("data-column") === colKey) el.classed("link-dimmed", false).classed("link-highlight", true);
            });
        d3.select(container)
            .selectAll("rect.gap-rect")
            .each(function() {
                const el = d3.select(this);
                if (el.attr("data-row") === rowKey || el.attr("data-column") === colKey) {
                    el.classed("gap-dimmed", false);
                }
            });
    }, [selectedNode, activeCell]);

    useEffect(() => {
        if (!showTooltips) setHoverTooltip(null);
    }, [showTooltips]);


    const clampedLinkMin = Math.max(0, Math.min(linkValueMin, linkValueMaxBound));
    const clampedLinkMax = Math.max(0, Math.min(linkValueMax, linkValueMaxBound));
    const normalizedLinkMin = Math.min(clampedLinkMin, clampedLinkMax);
    const normalizedLinkMax = Math.max(clampedLinkMin, clampedLinkMax);
    const linkStartPct = linkValueMaxBound > 0 ? (normalizedLinkMin / linkValueMaxBound) * 100 : 0;
    const linkEndPct = linkValueMaxBound > 0 ? (normalizedLinkMax / linkValueMaxBound) * 100 : 100;
    const normalizedEpochThresholdUi = Math.max(0, Math.min(maxEpochTransitionPct, epochTransitionThresholdUi));
    const epochPctPos = maxEpochTransitionPct > 0 ? (normalizedEpochThresholdUi / maxEpochTransitionPct) * 100 : 0;
    const shownEpochVisible = Math.max(0, Number(visibleEpochSummary.visible) || 0);
    const shownEpochTotal = Math.max(shownEpochVisible, Number(visibleEpochSummary.total) || 0);
    const commitEpochThreshold = (nextRaw: number) => {
        const next = Math.max(0, Math.min(maxEpochTransitionPct, Number(nextRaw) || 0));
        setEpochTransitionThresholdUi(next);
        if (Math.abs(epochTransitionThreshold - next) < 1e-6) return;
        setEpochTransitionThreshold(next);
        const p = new URLSearchParams(window.location.search);
        p.set("epochMin", next.toFixed(2));
        p.set("min", String(normalizedLinkMin));
        p.set("max", String(normalizedLinkMax));
        window.history.replaceState({}, "", window.location.pathname + "?" + p.toString());
    };
    const cellLegendMinText =
        selectedCellMetric === "cluster_count" ? cellLegendRange.min.toFixed(0) : cellLegendRange.min.toFixed(3);
    const cellLegendMaxText =
        selectedCellMetric === "cluster_count" ? cellLegendRange.max.toFixed(0) : cellLegendRange.max.toFixed(3);
    const cellMetricDisplayNameMap: Record<string, string> = {
        cluster_count: "Cluster Number",
        ari: "ARI",
        ami: "AMI",
        nmi: "NMI",
        homogeneity: "Homogeneity",
        completeness: "Completeness",
        v_measure: "V-Measure",
        fowlkes_mallows: "FMI",
    };
    const cellMetricDisplayName = cellMetricDisplayNameMap[selectedCellMetric] ?? selectedCellMetric;
    const displayCellInfo = hoveredCellInfo;
    const cellInfoLeft =
        displayCellInfo && layout
            ? transform.x +
            transform.scale * (layout.margin.left + displayCellInfo.col * layout.cellStepX + layout.cellW) +
            24
            : 0;
    const cellInfoTop =
        displayCellInfo && layout
            ? transform.y + transform.scale * (layout.margin.top + displayCellInfo.row * layout.cellStepY)
            : 0;

    return (
        <div className="timeline-root h-full flex flex-col" style={{ marginLeft: 0 }}>
            <div id="controls" className="shrink-0">
                <div className="controls-main">
                    <div className="control-stack">
                        <div className="control-row">
                            <span className="control-label">
                                <label>Link Range:</label>
                                <span
                                    className="value-range-help"
                                    data-tooltip="Filters visible Sankey links by link value. Links outside [min, max] are hidden."
                                    aria-label="Link value range help"
                                >
                                    ?
                                </span>
                            </span>
                            <span className="range-bound">{normalizedLinkMin.toFixed(0)}</span>
                            <div className="range-slider" id="rangeSlider">
                                <div
                                    className="range-track"
                                    id="rangeTrack"
                                    style={{
                                        background: `linear-gradient(to right, #ddd ${linkStartPct}%, #9c795b ${linkStartPct}%, #9c795b ${linkEndPct}%, #ddd ${linkEndPct}%)`,
                                    }}
                                ></div>
                                <input
                                    id="minRange"
                                    className="range-min"
                                    type="range"
                                    min={0}
                                    max={linkValueMaxBound}
                                    step={1}
                                    value={normalizedLinkMin}
                                    onChange={(e) => {
                                        const next = Number(e.target.value);
                                        const clamped = Math.max(0, Math.min(next, linkValueMaxBound));
                                        autoInitLinkMaxRef.current = false;
                                        setLinkValueMin(Math.min(clamped, normalizedLinkMax));
                                        const p = new URLSearchParams(window.location.search);
                                        p.set("min", String(clamped));
                                        p.set("max", String(normalizedLinkMax));
                                        p.set("epochMin", epochTransitionThreshold.toFixed(2));
                                        window.history.replaceState({}, "", window.location.pathname + "?" + p.toString());
                                    }}
                                />
                                <input
                                    id="maxRange"
                                    className="range-max"
                                    type="range"
                                    min={0}
                                    max={linkValueMaxBound}
                                    step={1}
                                    value={normalizedLinkMax}
                                    onChange={(e) => {
                                        const next = Number(e.target.value);
                                        const clamped = Math.max(0, Math.min(next, linkValueMaxBound));
                                        autoInitLinkMaxRef.current = false;
                                        setLinkValueMax(Math.max(clamped, normalizedLinkMin));
                                        const p = new URLSearchParams(window.location.search);
                                        p.set("min", String(normalizedLinkMin));
                                        p.set("max", String(clamped));
                                        p.set("epochMin", epochTransitionThreshold.toFixed(2));
                                        window.history.replaceState({}, "", window.location.pathname + "?" + p.toString());
                                    }}
                                />
                            </div>
                            <span className="range-bound">{normalizedLinkMax.toFixed(0)}</span>
                        </div>
                        <div className="control-row">
                            <span className="control-label">
                                <label>Epoch Min %:</label>
                                <span
                                    className="value-range-help"
                                    data-tooltip="For each epoch, we compute the maximum pct_changed across layer-to-layer transitions in that epoch. Epochs with max pct_changed below this threshold are hidden. The last epoch is always kept visible."
                                    aria-label="Epoch threshold help"
                                >
                                    ?
                                </span>
                            </span>
                            <span className="range-bound">0.00%</span>
                            <div className="epoch-slider-wrap">
                                <div className="range-slider" id="epochRangeSlider">
                                    <div
                                        className="range-track"
                                        style={{
                                            background: `linear-gradient(to right, #9c795b ${epochPctPos}%, #ddd ${epochPctPos}%)`,
                                        }}
                                    ></div>
                                    <input
                                        id="epochThresholdRange"
                                        className="range-single"
                                        type="range"
                                        min={0}
                                        max={maxEpochTransitionPct}
                                        step={0.1}
                                        value={normalizedEpochThresholdUi}
                                        onInput={(e) => {
                                            const next = Number((e.target as HTMLInputElement).value);
                                            commitEpochThreshold(next);
                                        }}
                                        onChange={(e) => {
                                            const next = Number(e.target.value);
                                            commitEpochThreshold(next);
                                        }}
                                        onPointerUp={(e) => commitEpochThreshold(Number((e.target as HTMLInputElement).value))}
                                        onMouseUp={(e) => commitEpochThreshold(Number((e.target as HTMLInputElement).value))}
                                        onTouchEnd={(e) => commitEpochThreshold(Number((e.target as HTMLInputElement).value))}
                                        onKeyUp={(e) => {
                                            if (
                                                e.key === "ArrowLeft" ||
                                                e.key === "ArrowRight" ||
                                                e.key === "ArrowUp" ||
                                                e.key === "ArrowDown" ||
                                                e.key === "Home" ||
                                                e.key === "End" ||
                                                e.key === "PageUp" ||
                                                e.key === "PageDown"
                                            ) {
                                                commitEpochThreshold(Number((e.target as HTMLInputElement).value));
                                            }
                                        }}
                                    />
                                </div>
                            </div>
                            <span id="epochThresholdValue" className="range-bound">
                                {normalizedEpochThresholdUi.toFixed(2)}%
                            </span>
                            <span className="epoch-visible-summary">
                                Shown: {shownEpochVisible}/{shownEpochTotal}
                            </span>
                        </div>
                    </div>
                    <div className="control-side">
                        <div id="controlButtons">
                            <button id="btnDefault">Default</button>
                            <button id="btnColumn">Column</button>
                            <button id="btnRow">Row</button>
                            <button id="btnSwap">Swap X/Y</button>
                            <button
                                type="button"
                                onClick={() => onToggleEpochOrder?.()}
                                style={{ fontWeight: reverseEpochOrder ? 600 : 400 }}
                                title="Flip epoch order"
                            >
                                Flip Epoch
                            </button>
                            <button
                                type="button"
                                onClick={() => setShowSankeyPaths((v) => !v)}
                                style={{ fontWeight: showSankeyPaths ? 600 : 400 }}
                            >
                                Paths
                            </button>
                            <button
                                type="button"
                                onClick={() => setShowSankeyNodes((v) => !v)}
                                style={{ fontWeight: showSankeyNodes ? 600 : 400 }}
                            >
                                Nodes
                            </button>
                            <button
                                type="button"
                                onClick={() => setShowCellBackground((v) => !v)}
                                style={{ fontWeight: showCellBackground ? 600 : 400 }}
                            >
                                {cellMetricDisplayName}
                            </button>
                            <button
                                type="button"
                                onClick={() => setShowGapHeatmap((v) => !v)}
                                style={{ fontWeight: showGapHeatmap ? 600 : 400 }}
                            >
                                Gap Heat
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setShowTooltips((v) => {
                                        const next = !v;
                                        if (!next) setHoverTooltip(null);
                                        return next;
                                    });
                                }}
                                style={{ fontWeight: showTooltips ? 600 : 400 }}
                            >
                                Tooltip
                            </button>
                        </div>
                    </div>
                </div>
                <div className="legend-bottom-row">
                    <span className="control-label legend-label">Legend:</span>
                    <div className="sankey-mini-legend" aria-label="Sankey Color Legend">
                        <span className="legend-item">
                            <span className="legend-text">{cellMetricDisplayName}</span>
                            <span className="legend-range">{cellLegendMinText}</span>
                            <span className="legend-swatch" style={{ background: "linear-gradient(to right, #eff6ff, #7B8FC7)" }} />
                            <span className="legend-range">{cellLegendMaxText}</span>
                        </span>
                        <span className="legend-item">
                            <span className="legend-text">Layer Change %</span>
                            <span className="legend-range">0</span>
                            <span className="legend-swatch" style={{ background: "linear-gradient(to right, #f3f4f6, #7f1d1d)" }} />
                            <span className="legend-range">{gapLegendMax.toFixed(1)}%</span>
                        </span>
                        <span className="legend-item">
                            <span className="legend-text">Epoch Change %</span>
                            <span className="legend-range">0</span>
                            <span className="legend-swatch" style={{ background: "linear-gradient(to right, #f0fdf4, #166534)" }} />
                            <span className="legend-range">{gapLegendMax.toFixed(1)}%</span>
                        </span>
                    </div>
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
                    <div ref={chartPanelRef} className="relative flex-1 min-w-0 pl-8">
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
                        {displayCellInfo && layout && (
                            <div
                                className="absolute z-20 pointer-events-none rounded border border-zinc-300 bg-white/95 px-1.5 py-0.5 text-[11px] text-zinc-700 shadow-sm whitespace-nowrap"
                                style={{
                                    left: cellInfoLeft,
                                    top: cellInfoTop,
                                    transform: "translate(-100%, 0)",
                                }}
                            >
                                {displayCellInfo.layerName} / {displayCellInfo.epochName}
                            </div>
                        )}
                        {hoverTooltip && (
                            <div
                                className="absolute z-30 pointer-events-none rounded border border-zinc-300 bg-white/95 px-2 py-1 text-[11px] text-zinc-700 shadow-sm whitespace-nowrap"
                                style={{
                                    left: hoverTooltip.x,
                                    top: hoverTooltip.y,
                                    transform: "translate(-50%, 8px)",
                                }}
                            >
                                <div className="tooltip-arrow" />
                                <div className="font-semibold text-zinc-800">{hoverTooltip.title}</div>
                                {hoverTooltip.lines.map((line, idx) => (
                                    <div key={`tt-${idx}`}>{line}</div>
                                ))}
                            </div>
                        )}

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
          margin: 4px 0;
          display: block;
          font-size: 13px;
        }

        .controls-main {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 6px;
          flex-wrap: wrap;
        }

        .control-stack {
          display: flex;
          flex-direction: column;
          gap: 4px;
          min-width: 470px;
        }
        
        .control-side {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 2px;
        }

        .control-row {
          display: flex;
          align-items: center;
          gap: 2px;
          flex-wrap: wrap;
        }
        
        .legend-label {
          min-width: 52px;
        }
        
        .legend-bottom-row {
          margin-top: 2px;
          display: flex;
          align-items: center;
          gap: 4px;
          flex-wrap: wrap;
        }

        .control-label {
          display: inline-flex;
          align-items: center;
          gap: 2px;
          min-width: 96px;
          font-size: 12px;
        }

        .range-bound {
          min-width: 30px;
          display: inline-block;
          text-align: right;
          font-variant-numeric: tabular-nums;
          font-size: 12px;
        }

        #chart {
          width: fit-content;
          max-width: none;
          margin: 0;
        }

        :global(.link-path.hidden) {
          display: none;
        }

        .range-slider {
          position: relative;
          width: 170px;
          height: 24px;
          display: inline-block;
          align-items: center;
        }
        
        .epoch-slider-wrap {
          display: inline-flex;
        }

        .epoch-visible-summary {
          font-size: 12px;
          color: #475569;
          line-height: 1.2;
          white-space: nowrap;
        }

        #controlButtons {
          display: grid;
          grid-template-columns: repeat(5, max-content);
          gap: 2px;
          width: auto;
          margin-top: 0;
        }
        
        .sankey-mini-legend {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 11px;
          color: #475569;
          line-height: 1;
          white-space: nowrap;
        }

        .legend-item {
          display: inline-flex;
          align-items: center;
          gap: 4px;
        }

        .legend-text {
          min-width: 42px;
          text-align: right;
        }

        .legend-range {
          min-width: 26px;
          text-align: center;
          font-variant-numeric: tabular-nums;
        }

        .legend-swatch {
          width: 84px;
          height: 8px;
          border-radius: 4px;
          border: 1px solid rgba(15, 23, 42, 0.25);
          display: inline-block;
        }

        .tooltip-arrow {
          position: absolute;
          left: 50%;
          top: -6px;
          width: 0;
          height: 0;
          transform: translateX(-50%);
          border-left: 6px solid transparent;
          border-right: 6px solid transparent;
          border-bottom: 6px solid rgba(255, 255, 255, 0.95);
          filter: drop-shadow(0 -1px 0 rgba(148, 163, 184, 0.9));
        }

        #controlButtons button {
          font-size: 12px;
          padding: 1px 4px;
          margin: 0;
          line-height: 1.1;
          white-space: nowrap;
        }

        .range-slider input[type="range"] {
          position: absolute;
          left: 0;
          top: 7px;
          width: 100%;
          pointer-events: none;
          touch-action: none;
          -webkit-appearance: none;
          background: transparent;
        }

        .range-slider input.range-min {
          z-index: 3;
        }

        .range-slider input.range-max {
          z-index: 2;
        }

        .range-slider input.range-single {
          z-index: 3;
        }

        .range-slider input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          pointer-events: auto;
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background: #fff;
          border: 1.5px solid #666;
          box-shadow: 0 0 2px rgba(0, 0, 0, 0.3);
        }

        .range-slider input[type="range"]::-moz-range-thumb {
          pointer-events: auto;
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background: #fff;
          border: 1.5px solid #666;
          box-shadow: 0 0 2px rgba(0, 0, 0, 0.3);
        }

        .range-slider input[type="range"]::-moz-range-track {
          pointer-events: none;
          background: transparent;
          border: 0;
        }

        .range-track {
          position: absolute;
          left: 0;
          top: 10px;
          height: 4px;
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

        :global(.gap-dimmed) {
          opacity: 0.12 !important;
        }

        :global(#chart svg rect.bar.node-highlight) {
          stroke: #ef4444 !important;
          stroke-width: 2 !important;
          opacity: 1 !important;
        }

        :global(#chart svg rect.bar.node-context) {
          opacity: 1 !important;
        }

        :global(#chart svg rect.bar.node-dimmed) {
          opacity: 0.12 !important;
        }

        :global(#chart svg path.link-path) {
          pointer-events: none;
        }
      `}</style>
        </div>
    );
}
