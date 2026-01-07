"use client";
import * as d3 from "d3";
import { brush } from "d3-brush";
import React, {
    useEffect,
    useRef,
    forwardRef,
    useImperativeHandle,
} from "react";

interface ScatterPlotProps {
    onScroll?: (scrollLeft: number) => void;

    data?: {
        x: number;
        y: number;
        real_id: string | number;
        layer: string;
        color_value?: number;
        id?: string;
    }[];

    clusterFilePath?: string;

    totalWindows?: number;
    zoomFactor?: number;

    hideControls?: boolean;
}

const ScatterPlot = forwardRef<HTMLDivElement, ScatterPlotProps>(
    ({ onScroll, data = [], totalWindows, zoomFactor = 1, hideControls = false }, ref) => {
        let tooltip = d3.select("body").select("#global-tooltip");
        if (tooltip.empty()) {
            d3.select("body")
                .append("div")
                .attr("id", "global-tooltip")
                .style("position", "absolute")
                .style("padding", "4px")
                .style("background", "white")
                .style("border", "1px solid #ccc")
                .style("borderRadius", "4px")
                .style("pointerEvents", "none")
                .style("fill", "black")
                .style("zIndex", "9999")
                .style("opacity", 0);
        }

        // Always reset style every mount (keeps tooltip consistent)
        tooltip
            .style("position", "absolute")
            .style("padding", "4px")
            .style("background", "white")
            .style("fill", "black")
            .style("border", "1px solid #ccc")
            .style("borderRadius", "4px")
            .style("pointerEvents", "none")
            .style("zIndex", "9999")
            .style("opacity", 0)
            .html("");

        const containerRef = useRef<HTMLDivElement | null>(null);
        // Move selectedIDs to top-level so it's accessible in clearScatter and run
        let selectedIDs = new Set<string>();
        const [lassoEnabled, setLassoEnabled] = React.useState(false);

        // Step 1 helper: select points only inside current container
        const selectLocalPoints = () =>
            d3.select(containerRef.current).selectAll<SVGCircleElement, any>(".scatter-point");

        useEffect(() => {
            // Passive, stateless highlight from Sankey or other sources
            const handleScatterHighlight = (e: any) => {
                const ids = e?.detail?.ids ?? e?.detail;
                const source = e?.detail?.source;

                if (!Array.isArray(ids)) return;
                if (source === "scatter") return;

                selectLocalPoints()
                    .style("opacity", 0.1)
                    .attr("r", 4);

                ids.forEach((id: string) => {
                    selectLocalPoints()
                        .filter(`.real-${id}`)
                        .style("opacity", 1)
                        .attr("r", 6);
                });
            };

            const clearScatter = () => {
                selectedIDs.clear?.(); // if exists
                selectLocalPoints()
                    .style("opacity", 0.8)
                    .attr("r", 4);
                // safely clear active brushes only when lasso is enabled
                if (lassoEnabled) {
                    d3.select(containerRef.current)
                        .selectAll(".brush")
                        .each(function() {
                            // cast to any so .call() overload checking doesn't choke on d3's complex types
                            const brushG = d3.select(this) as any;
                            const state = (brushG.node() as any).__brush;
                            if (state) {
                                // call the brush move — keep runtime semantics, silence TypeScript overload mismatch
                                brushG.call(d3.brush().move as any, null);
                            }
                        });
                }
                window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
            };

            window.addEventListener("scatter-highlight-ids", handleScatterHighlight);
            window.addEventListener("scatter-highlight-clear", clearScatter);

            return () => {
                window.removeEventListener("scatter-highlight-ids", handleScatterHighlight);
                window.removeEventListener("scatter-highlight-clear", clearScatter);
            };
        }, [lassoEnabled]);

        useImperativeHandle(ref, () => containerRef.current as HTMLDivElement);

        // NEW: zoom handler — rescale wrapper widths without redrawing circles
        useEffect(() => {
            const container = containerRef.current;
            if (!container) return;

            // select all wrappers
            d3.select(container)
                .selectAll<HTMLDivElement, unknown>("div.scatter-wrapper")
                .style(
                    "width",
                    function(this: HTMLDivElement, d: unknown, i: number, groups: ArrayLike<HTMLDivElement>) {
                        const node = groups[i] as HTMLDivElement;
                        const baseWidth = (node as any).__baseWidth || node.clientWidth / zoomFactor;
                        return baseWidth * zoomFactor + "px";
                    }
                );

            // scale SVGs too
            d3.select(container)
                .selectAll<SVGSVGElement, unknown>("svg.scatter-svg")
                .attr("width", function(this: SVGSVGElement, d: unknown, i: number, groups: ArrayLike<SVGSVGElement>) {
                    const svg = d3.select(this);
                    const base = (svg as any).node().__baseWidth || +svg.attr("width") / zoomFactor;
                    return base * zoomFactor; // number is accepted
                });
        }, [zoomFactor]);

        // Prevent StrictMode double-run
        const hasDrawnRef = React.useRef(false);

        useEffect(() => {
            const container = containerRef.current;
            if (!container) return;

            d3.select(container).selectAll("*").remove();
            hasDrawnRef.current = false;

            const run = async () => {
                // Use layerFiles from data, not hardcoded
                const layerFiles = Array.from(
                    new Set(data.map(d => `${d.layer}.csv`))
                );
                window.dispatchEvent(
                    new CustomEvent("scatter-layer-info", {
                        detail: {
                            windowId: undefined,
                            names: layerFiles,
                            count: layerFiles.length
                        }
                    })
                );

                const height = 200;
                // selectedIDs now defined at component top-level

                for (let i = 0; i < layerFiles.length; i++) {
                    // subtract total margin (5px per gap)
                    const totalMargin = (layerFiles.length - 1) * 5;
                    const availableWidth = container.clientWidth - totalMargin;
                    const svgWidth = availableWidth / layerFiles.length;

                    const wrapper = d3
                        .select(container)
                        .append("div")
                        .classed("scatter-wrapper", true)
                        .each(function() { (this as any).__baseWidth = svgWidth; })
                        .style("width", (svgWidth * zoomFactor) + "px")
                        .style("display", "flex")
                        .style("flex-direction", "column")
                        .style("align-items", "center")
                        .style("margin-right", "5px");

                    wrapper
                        .append("div")
                        .style("font-size", "12px")
                        .style("margin-bottom", "4px")
                        .text(layerFiles[i].replace(".csv", ""));

                    const svg = wrapper
                        .append("svg")
                        .classed("scatter-svg", true)
                        .attr("height", height)
                        .each(function() { (this as any).__baseWidth = svgWidth; })
                        .attr("width", svgWidth * zoomFactor)
                        .style("background", "#fafafa");

                    // Instead of reading CSV, filter from data
                    const layerName = layerFiles[i].replace(".csv", "");
                    const localData = data
                        .filter(d => d.layer === layerName)
                        .map(d => ({
                            x: d.x,
                            y: d.y,
                            real_id: d.id,
                            color_value: d.color_value,
                        }));
                    console.log("data", data);
                    console.log("localdata", localData);

                    const xExtent = d3.extent(localData, d => d.x) as [number, number];
                    const yExtent = d3.extent(localData, d => d.y) as [number, number];

                    const xScale = d3.scaleLinear()
                        .domain(xExtent)
                        .range([10, svgWidth - 10]);

                    const yScale = d3.scaleLinear()
                        .domain(yExtent)
                        .range([height - 10, 10]); // invert y

                    const color = d3.schemeCategory10[i % 10];
                    // Create group for circles so we can attach events
                    const circleGroup = svg.append("g");

                    const circles = circleGroup
                        .selectAll("circle")
                        .data(localData)
                        .join("circle")
                        .attr("cx", (d) => xScale(d.x))
                        .attr("cy", (d) => yScale(d.y))
                        .attr("r", 4)
                        .attr("fill", (d) => {
                            return d.color_value !== undefined && !isNaN(d.color_value)
                                ? d3.interpolateTurbo(d.color_value / 3)
                                : color;
                        })
                        .attr("opacity", 0.8)
                        .attr("class", (d) =>
                            d.real_id !== undefined ? `scatter-point real-${d.real_id}` : "scatter-point"
                        );
                    // Add hover events to dim others and highlight same real_id across all plots
                    circles
                        .on("mouseover", function(event, d) {
                            if (lassoEnabled) return;

                            const idClass = `real-${d.real_id}`;
                            console.log("Hover ID:", d.real_id);
                            selectLocalPoints().style("opacity", 0.1);

                            selectLocalPoints()
                                .filter(`.${idClass}`)
                                .style("opacity", 1)
                                .attr("r", 6);

                            if (!hideControls) {
                                tooltip
                                    .html(`<p>TEST ${d.real_id}</p>`)
                                    .style("left", (event.pageX + 10) + "px")
                                    .style("top", (event.pageY + 10) + "px")
                                    .style("opacity", 1);
                            }
                        })
                        .on("mouseout", function() {
                            if (lassoEnabled) return;

                            selectLocalPoints()
                                .style("opacity", 0.8)
                                .attr("r", 4);
                            if (!hideControls) {
                                tooltip.style("opacity", 0);
                            }
                        });

                    // --- LASSO / BRUSH SELECTION ---
                    const brushLayer = svg.append("g").attr("class", "brush");

                    const brushed = (event: any) => {
                        const selection = event.selection;
                        if (!selection) {
                            selectedIDs.clear();
                            selectLocalPoints()
                                .style("opacity", 0.8)
                                .attr("r", 4);
                            window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
                            return;
                        }

                        const [[x0, y0], [x1, y1]] = selection;
                        selectedIDs.clear();

                        circles.each(function(d: any) {
                            const cx = xScale(d.x);
                            const cy = yScale(d.y);
                            if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) {
                                selectedIDs.add(d.real_id);
                            }
                        });

                        selectLocalPoints()
                            .style("opacity", 0.1)
                            .attr("r", 4);

                        selectedIDs.forEach(id => {
                            selectLocalPoints()
                                .filter(`.real-${id}`)
                                .style("opacity", 1)
                                .attr("r", 6);
                        });

                        window.dispatchEvent(
                            new CustomEvent("sankey-highlight-ids", {
                                detail: { ids: Array.from(selectedIDs), source: "scatter" },
                            })
                        );
                    };

                    if (lassoEnabled) {
                        brushLayer.call(
                            d3.brush()
                                .extent([[0, 0], [svgWidth, height]])
                                .on("brush end", brushed)
                        );
                    } else {
                        brushLayer.selectAll("*").remove();
                    }
                }
            };

            run();
        }, [data, zoomFactor, lassoEnabled]);

        return (
            <div>
                {!hideControls && (
                    <>
                        <div style={{
                            display: "flex",
                            flexDirection: "row",
                            gap: "1rem",
                            alignItems: "center",
                            padding: "0.5rem 0",
                            borderBottom: "1px solid #ddd",
                            marginBottom: "0.5rem"
                        }}>
                            <div>
                                <label style={{ fontSize: "0.8rem", marginRight: "0.5rem" }}>Dimensionality Reduction</label>
                                <select
                                    style={{ padding: "2px 6px", fontSize: "0.8rem" }}
                                    onChange={(e) => {
                                        window.dispatchEvent(
                                            new CustomEvent("scatter-method-change", {
                                                detail: e.target.value
                                            })
                                        );
                                    }}
                                >
                                    <option value="umap">UMAP</option>
                                    <option value="tsne">t-SNE</option>
                                    <option value="pca">PCA</option>
                                    <option value="lle">LLE</option>
                                    <option value="isomap">Isomap</option>
                                </select>
                            </div>

                            <div>
                                <label style={{ fontSize: "0.8rem", marginRight: "0.5rem" }}>Color Legend</label>
                                <select
                                    style={{ padding: "2px 6px", fontSize: "0.8rem" }}
                                    onChange={(e) => {
                                        window.dispatchEvent(
                                            new CustomEvent("scatter-color-legend-change", {
                                                detail: e.target.value
                                            })
                                        );
                                    }}
                                >
                                    <option value="cluster">Cluster</option>
                                    <option value="density">Density</option>
                                </select>
                            </div>
                        </div>
                    </>
                )}

                <button
                    onClick={() => setLassoEnabled(!lassoEnabled)}
                    style={{
                        padding: "4px 8px",
                        fontSize: "0.8rem",
                        background: lassoEnabled ? "#c62828" : "#2e7d32",
                        color: "white",
                        borderRadius: "4px",
                        marginLeft: "1rem",
                    }}
                >
                    {lassoEnabled ? "Disable Lasso" : "Enable Lasso"}
                </button>

                <div
                    ref={containerRef}
                    onScroll={(e) => {
                        const left = (e.target as HTMLDivElement).scrollLeft;
                        onScroll?.(left);
                    }}
                    onWheel={(e) => {
                        const container = e.currentTarget as HTMLDivElement;
                        container.scrollLeft += e.deltaY;
                        // Notify WindowManager for global synchronization
                        window.dispatchEvent(
                            new CustomEvent("sankey-scroll", { detail: container.scrollLeft })
                        );
                    }}
                    style={{
                        display: "flex",
                        flexDirection: "row",
                        width: "100%",
                        overflowX: "scroll",
                        overflowY: "hidden",
                        paddingBottom: "0.5rem"
                    }}
                ></div>
            </div>
        );
    }
);

export default ScatterPlot;
