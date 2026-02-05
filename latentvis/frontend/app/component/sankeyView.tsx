"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import ZoomPan from "./ZoomPan.tsx";

const DATA_PATH = "/data/sankey/all_sankey.json";
const DEFAULT_MAX_INSTANCE = 158;

export default function SankeyView() {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [maxInstanceValue, setMaxInstanceValue] = useState(DEFAULT_MAX_INSTANCE);
  const [layout, setLayout] = useState<{
    rowNames: string[];
    colNames: string[];
    cellW: number;
    cellH: number;
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

    fetch(DATA_PATH)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch all_sankey.json");
        return res.json();
      })
      .then((data) => {
        const ROW_NAMES: string[] = Object.keys(data?.epoch || {});
        const COLUMN_NAMES: string[] = Object.keys(data?.layer || {});

        const ROW_INDEX = new Map(ROW_NAMES.map((d, i) => [d, i]));
        const COLUMN_INDEX = new Map(COLUMN_NAMES.map((d, i) => [d, i]));

        const X_NAMES = SWAP_AXES ? ROW_NAMES : COLUMN_NAMES;
        const Y_NAMES = SWAP_AXES ? COLUMN_NAMES : ROW_NAMES;

        function cellOrigin(rowName: string, columnName: string, cellW: number, cellH: number) {
          const col = SWAP_AXES ? (ROW_INDEX.get(rowName) ?? 0) : (COLUMN_INDEX.get(columnName) ?? 0);
          const row = SWAP_AXES ? (COLUMN_INDEX.get(columnName) ?? 0) : (ROW_INDEX.get(rowName) ?? 0);
          return { col, row, x: col * cellW, y: row * cellH };
        }

        const margin = { top: 20, left: 30, right: 0, bottom: 20 };
        const CELL_W = 120;
        const CELL_H = 150;

        const width = margin.left + margin.right + CELL_W * X_NAMES.length;
        const height = margin.top + margin.bottom + CELL_H * Y_NAMES.length;

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
          innerW,
          innerH,
          margin,
          cellMargin,
        });

        const cells: d3.Selection<SVGGElement, any, any, any>[] = [];
        X_NAMES.forEach((xKey, col) => {
          Y_NAMES.forEach((yKey, row) => {
            const rowName = SWAP_AXES ? xKey : yKey;
            const columnName = SWAP_AXES ? yKey : xKey;

            const g = gRoot
              .append("g")
              .attr("id", `cell-${rowName}-${columnName}`)
              .attr("transform", `translate(${col * cellW}, ${row * cellH})`)
              .datum({ rowName, columnName, col, row });

            g.append("rect").attr("width", cellW).attr("height", cellH).attr("fill", "white").attr("stroke", "#888");

            const innerG = g.append("g").attr("transform", `translate(${cellMargin.left}, ${cellMargin.top})`);
            cells.push(innerG);

            g.on("mouseenter", (_event: any, datum: any) => {
              setCellHighlight(datum.rowName, datum.columnName, true);
            }).on("mouseleave", (_event: any, datum: any) => {
              setCellHighlight(datum.rowName, datum.columnName, false);
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

        const x = d3.scaleBand().range([0, innerW]).domain(d3.range(maxClusters) as any).padding(0.2);
        const y = d3.scaleLinear().domain([0, maxInstance]).range([innerH, 0]);

        const nodeLookup = new Map<string, { rowName: string; columnName: string; idx: number; node: any }>();

        function columnFromLabel(label: string) {
          const i = label.lastIndexOf("_C");
          if (i === -1) return label;
          return label.slice(0, i);
        }

        cells.forEach((g) => {
          const { rowName, columnName } = g.datum();
          const rowData = data.epoch?.[rowName];
          const nodesForCell = (rowData && rowData.nodes && rowData.nodes[columnName]) || [];
          g.selectAll("rect.bar")
            .data(nodesForCell)
            .enter()
            .append("rect")
            .attr("class", "bar")
            .attr("x", (_d: any, i: number) => x(i) as number)
            .attr("y", (d: any) => innerH - y(d.previous))
            .attr("width", x.bandwidth())
            .attr("height", (d: any) => innerH - y(d.value))
            .attr("fill", "#aaa");

          nodesForCell.forEach((node: any, idx: number) => {
            const label = String(node.label ?? "");
            const key = `${rowName}|${columnName}|${label}`;
            nodeLookup.set(key, { rowName, columnName, idx, node });
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

        // column-links (per layer) -> vertical inside a column
        const linkGroup = gRoot.append("g").attr("class", "links column-links");
        Object.keys(data.layer || {}).forEach((columnKey) => {
          const columnObj = data.layer[columnKey];
          if (COLUMN_INDEX.get(columnKey) == null) return;

          (columnObj.edges || []).forEach((edge: any) => {
            let sourceInfo: any = null;
            let targetInfo: any = null;
            const nodesByRow = columnObj.nodes || {};

            Object.keys(nodesByRow).forEach((rowName) => {
              (nodesByRow[rowName] || []).forEach((node: any, idx: number) => {
                if (node.id === edge.source_node_id) sourceInfo = { node, rowName, idx };
                if (node.id === edge.target_node_id) targetInfo = { node, rowName, idx };
              });
            });

            if (!sourceInfo || !targetInfo) return;

            const srcO = cellOrigin(sourceInfo.rowName, columnKey, cellW, cellH);
            const tgtO = cellOrigin(targetInfo.rowName, columnKey, cellW, cellH);

            const srcIdx = sourceInfo.idx;
            const tgtIdx = targetInfo.idx;
            const srcBand = x(srcIdx);
            const tgtBand = x(tgtIdx);
            if (srcBand == null || tgtBand == null) return;

            const sPrev = sourceInfo.node.previous ?? 0;
            const sVal = sourceInfo.node.value ?? 0;
            const tPrev = targetInfo.node.previous ?? 0;
            const tVal = targetInfo.node.value ?? 0;

            const sBarY = innerH - y(sPrev);
            const sBarH = innerH - y(sVal);
            const tBarY = innerH - y(tPrev);
            const tBarH = innerH - y(tVal);

            const strokeW = linkWidthScale(+edge.value || 0);

            let pathD = "";
            if (!SWAP_AXES) {
              const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + x.bandwidth() / 2;
              const srcCenterY = srcO.y + cellMargin.top + sBarY + Math.max(0, sBarH);
              const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number) + x.bandwidth() / 2;
              const tgtCenterY = tgtO.y + cellMargin.top + tBarY;
              const dy = Math.abs(tgtCenterY - srcCenterY) * 0.5;
              pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX},${srcCenterY + dy}
                       ${tgtCenterX},${tgtCenterY - dy}
                       ${tgtCenterX},${tgtCenterY}`;
            } else {
              const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + x.bandwidth();
              const srcCenterY = srcO.y + cellMargin.top + (sBarY + Math.max(0, sBarH) / 2);
              const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number);
              const tgtCenterY = tgtO.y + cellMargin.top + (tBarY + Math.max(0, tBarH) / 2);
              const dx = Math.abs(tgtCenterX - srcCenterX) * 0.5;
              pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX + dx},${srcCenterY}
                       ${tgtCenterX - dx},${tgtCenterY}
                       ${tgtCenterX},${tgtCenterY}`;
            }

            const p = appendLinkPath(linkGroup as any, pathD, edge.value, strokeW, "#86A69D", 0.3);
            p.attr("data-column", columnKey);
          });
        });

        // row-links (per epoch/model) -> horizontal inside a row
        const linkGroup2 = gRoot.append("g").attr("class", "links row-links");
        Object.keys(data.epoch || {}).forEach((rowKey) => {
          const rowObj = data.epoch[rowKey];
          if (ROW_INDEX.get(rowKey) == null) return;

          (rowObj.edges || []).forEach((edge: any) => {
            const sLabel = String(edge.source_label ?? "");
            const tLabel = String(edge.target_label ?? "");
            const sCol = columnFromLabel(sLabel);
            const tCol = columnFromLabel(tLabel);
            const sourceInfo = nodeLookup.get(`${rowKey}|${sCol}|${sLabel}`) || null;
            const targetInfo = nodeLookup.get(`${rowKey}|${tCol}|${tLabel}`) || null;

            if (!sourceInfo || !targetInfo) return;

            const srcO = cellOrigin(rowKey, sourceInfo.columnName, cellW, cellH);
            const tgtO = cellOrigin(rowKey, targetInfo.columnName, cellW, cellH);

            const srcIdx = sourceInfo.idx;
            const tgtIdx = targetInfo.idx;
            const srcBand = x(srcIdx);
            const tgtBand = x(tgtIdx);
            if (srcBand == null || tgtBand == null) return;

            const sPrev = sourceInfo.node.previous ?? 0;
            const sVal = sourceInfo.node.value ?? 0;
            const tPrev = targetInfo.node.previous ?? 0;
            const tVal = targetInfo.node.value ?? 0;

            const sBarY = innerH - y(sPrev);
            const sBarH = innerH - y(sVal);
            const tBarY = innerH - y(tPrev);
            const tBarH = innerH - y(tVal);

            const strokeW = linkWidthScale(+edge.value || 0);

            let pathD = "";
            if (!SWAP_AXES) {
              const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + x.bandwidth();
              const srcCenterY = srcO.y + cellMargin.top + (sBarY + Math.max(0, sBarH) / 2);
              const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number);
              const tgtCenterY = tgtO.y + cellMargin.top + (tBarY + Math.max(0, tBarH) / 2);
              const dx = Math.abs(tgtCenterX - srcCenterX) * 0.5;
              pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX + dx},${srcCenterY}
                       ${tgtCenterX - dx},${tgtCenterY}
                       ${tgtCenterX},${tgtCenterY}`;
            } else {
              const srcCenterX = srcO.x + cellMargin.left + (srcBand as number) + x.bandwidth() / 2;
              const srcCenterY = srcO.y + cellMargin.top + sBarY + Math.max(0, sBarH);
              const tgtCenterX = tgtO.x + cellMargin.left + (tgtBand as number) + x.bandwidth() / 2;
              const tgtCenterY = tgtO.y + cellMargin.top + tBarY;
              const dy = Math.abs(tgtCenterY - srcCenterY) * 0.5;
              pathD = `M ${srcCenterX},${srcCenterY}
                     C ${srcCenterX},${srcCenterY + dy}
                       ${tgtCenterX},${tgtCenterY - dy}
                       ${tgtCenterX},${tgtCenterY}`;
            }

            const p2 = appendLinkPath(linkGroup2 as any, pathD, edge.value, strokeW, "#F28585", 0.2);
            p2.attr("data-row", rowKey);
          });
        });

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
            track.style.background = `linear-gradient(to right, #ddd ${startPct}%, #86A69D ${startPct}%, #86A69D ${endPct}%, #ddd ${endPct}%)`;
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
  }, []);

  return (
    <div className="timeline-root h-full flex flex-col">
      <div id="controls" className="shrink-0">
        <label style={{ marginRight: 8 }}>Value range:</label>
        <div className="range-slider" id="rangeSlider">
          <div className="range-track" id="rangeTrack"></div>
          <input id="minRange" className="range-min" type="range" min={0} max={maxInstanceValue} step={1} defaultValue={0} />
          <input id="maxRange" className="range-max" type="range" min={0} max={maxInstanceValue} step={1} defaultValue={maxInstanceValue} />
        </div>
        <span id="minValue" style={{ marginLeft: 12 }}>0</span>
        <span id="sep" style={{ marginLeft: 6, marginRight: 6 }}>—</span>
        <span id="maxValue">{maxInstanceValue}</span>

        <small style={{ color: "#666", marginLeft: 12 }}>
          link value threshold — Mode: Default, Column, Row
        </small>
        <div style={{ marginLeft: 0, display: "flex", gap: 8, alignItems: "center" }}>
          <button id="btnDefault">Default</button>
          <button id="btnColumn">Column</button>
          <button id="btnRow">Row</button>
          <button id="btnSwap">Swap X/Y</button>
        </div>
      </div>

      <div className="relative w-full flex-1 min-h-0 overflow-hidden">
        <ZoomPan className="w-full h-full" onTransform={setTransform}>
          <div id="chart" ref={chartRef}></div>
        </ZoomPan>

        {layout && (
          <>
            <div className="absolute left-0 top-0 right-0 h-6 pointer-events-none bg-white/90 z-10">
              {layout.colNames.map((name, i) => {
                const x =
                  transform.x +
                  transform.scale * (layout.margin.left + i * layout.cellW + layout.cellMargin.left + layout.innerW / 2);
                return (
                  <div
                    key={name}
                    className="absolute text-xs text-zinc-700"
                    style={{ transform: `translate(${x}px, 0px) translateX(-50%)` }}
                  >
                    {name}
                  </div>
                );
              })}
            </div>
            <div className="absolute left-0 top-0 bottom-0 w-6 pointer-events-none bg-white/90 z-10">
              {layout.rowNames.map((name, i) => {
                const y =
                  transform.y +
                  transform.scale * (layout.margin.top + i * layout.cellH + layout.cellMargin.top + layout.innerH / 2);
                return (
                  <div
                    key={name}
                    className="absolute text-xs text-zinc-700"
                    style={{ transform: `translate(-27px, ${y}px) translateY(-50%) rotate(-90deg)` }}
                  >
                    {name}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>

      <style jsx>{`
        .timeline-root {
          margin: 0;
          font-family: system-ui, -apple-system, BlinkMacSystemFont;
          background: #fff;
        }

        #controls {
          width: fit-content;
          max-width: 100%;
          margin: 8px auto;
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 14px;
          flex-wrap: wrap;
        }

        #chart {
          width: fit-content;
          max-width: 100%;
          margin: 10px auto;
        }

        input[type="range"] {
          width: 600px;
        }

        :global(.link-path.hidden) {
          display: none;
        }

        .range-slider {
          position: relative;
          width: 600px;
          height: 36px;
          display: inline-block;
          align-items: center;
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

        .range-track {
          position: absolute;
          left: 0;
          top: 18px;
          height: 6px;
          width: 100%;
          background: linear-gradient(to right, #ddd 0%, #ddd 100%);
          border-radius: 3px;
        }

        :global(.cell-dimmed) {
          opacity: 0.12 !important;
        }

        :global(.link-dimmed) {
          stroke-opacity: 0.06 !important;
        }
      `}</style>
    </div>
  );
}
