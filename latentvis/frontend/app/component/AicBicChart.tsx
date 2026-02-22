"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

type AicBicData = {
  model: string;
  layer: string;
  k_values: number[];
  aic: number[];
  bic: number[];
};

type AicBicChartProps = {
  dataPath: string;
  allDataPaths?: string[];
  isSelected?: boolean;
};

export default function AicBicChart({ dataPath, allDataPaths, isSelected }: AicBicChartProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const [activeMetric, setActiveMetric] = useState<"aic" | "bic" | "both">("both");
  const [kMin, setKMin] = useState(0);
  const [kMax, setKMax] = useState(1);
  const [rangeMin, setRangeMin] = useState(0);
  const [rangeMax, setRangeMax] = useState(1);

  useEffect(() => {
    const container = chartRef.current;
    if (!container) return;
    container.innerHTML = "";
    container.style.position = "relative";

    const ro = new ResizeObserver(() => draw());
    ro.observe(container);

    let lastData: AicBicData | null = null;
    let allData: AicBicData[] | null = null;

    function draw() {
      if (!container) return;
      container.innerHTML = "";
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);

      const margin = { top: 10, right: 22, bottom: 16, left: 22 };
      const innerW = Math.max(1, width - margin.left - margin.right);
      const innerH = Math.max(1, height - margin.top - margin.bottom);

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      const dataToRender = allData ?? (lastData ? [lastData] : []);
      if (!dataToRender.length) return;

      const kValues = dataToRender[0].k_values;
      if (!kValues?.length) return;
      const kMinVal = d3.min(kValues) ?? 0;
      const kMaxVal = d3.max(kValues) ?? 1;
      const effectiveMin = Math.max(kMinVal, Math.min(rangeMin, kMaxVal));
      const effectiveMax = Math.max(kMinVal, Math.min(rangeMax, kMaxVal));

      const aicValues = dataToRender.flatMap((d) => d.aic ?? []);
      const bicValues = dataToRender.flatMap((d) => d.bic ?? []);
      if (!aicValues.length || !bicValues.length) return;

      const x = d3
        .scaleLinear()
        .domain([d3.min(kValues) ?? 0, d3.max(kValues) ?? 1])
        .range([0, innerW]);

      const yAic = d3.scaleLinear().domain([0, 1]).range([innerH, 0]).nice();
      const yBic = d3.scaleLinear().domain([0, 1]).range([innerH, 0]).nice();

      const aicColor = "#7A83BF";
      const bicColor = "#B5BF4F";
      const showAic = activeMetric === "aic" || activeMetric === "both";
      const showBic = activeMetric === "bic" || activeMetric === "both";
      g.append("g")
        .attr("transform", `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(Math.min(4, kValues.length)).tickSizeOuter(0))
        .call((sel) => sel.selectAll("text").attr("font-size", 9));

      g.append("g")
        .call(d3.axisLeft(yAic).ticks(2).tickSizeOuter(0))
        .call((sel) => sel.selectAll("text").attr("font-size", 9).attr("fill", aicColor));

      g.append("g")
        .attr("transform", `translate(${innerW},0)`)
        .call(d3.axisRight(yBic).ticks(2).tickSizeOuter(0))
        .call((sel) => sel.selectAll("text").attr("font-size", 9).attr("fill", bicColor));

      const aicLine = d3
        .line<number>()
        .x((_, i) => x(kValues[i]))
        .y((d) => yAic(d))
        .curve(d3.curveMonotoneX);

      const bicLine = d3
        .line<number>()
        .x((_, i) => x(kValues[i]))
        .y((d) => yBic(d))
        .curve(d3.curveMonotoneX);

      const baseOpacity = isSelected ? 0.35 : 0.8;
      const baseWidth = isSelected ? 2.6 : 2.1;

      dataToRender.forEach((d) => {
        if (!d.k_values?.length || !d.aic?.length || !d.bic?.length) return;
        const isActive = isSelected && lastData && d.model === lastData.model && d.layer === lastData.layer;
        const opacity = isSelected ? (isActive ? 1 : 0.2) : baseOpacity;
        const width = isSelected ? (isActive ? 2.6 : baseWidth) : baseWidth;
        const aicOpacity = showAic ? opacity : 0.08;
        const bicOpacity = showBic ? opacity : 0.08;
        const aicWidth = showAic ? width : Math.max(0.9, width - 0.7);
        const bicWidth = showBic ? width : Math.max(0.9, width - 0.7);
        const aicMin = d3.min(d.aic) ?? 0;
        const aicMax = d3.max(d.aic) ?? 1;
        const aicNorm = d.aic.map((v) => (aicMax === aicMin ? 0.5 : (v - aicMin) / (aicMax - aicMin)));
        const bicMin = d3.min(d.bic) ?? 0;
        const bicMax = d3.max(d.bic) ?? 1;
        const bicNorm = d.bic.map((v) => (bicMax === bicMin ? 0.5 : (v - bicMin) / (bicMax - bicMin)));

        g.append("path")
          .datum(aicNorm)
          .attr("d", aicLine)
          .attr("fill", "none")
          .attr("stroke", aicColor)
          .attr("stroke-width", aicWidth)
          .attr("stroke-opacity", aicOpacity);

        g.append("path")
          .datum(bicNorm)
          .attr("d", bicLine)
          .attr("fill", "none")
          .attr("stroke", bicColor)
          .attr("stroke-width", bicWidth)
          .attr("stroke-opacity", bicOpacity);
      });

      const brush = d3
        .brushX()
        .extent([
          [0, 0],
          [innerW, innerH],
        ])
        .on("end", (event) => {
          const sel = event.selection as [number, number] | null;
          if (!sel) return;
          const invMin = x.invert(sel[0]);
          const invMax = x.invert(sel[1]);
          const kLo = Math.max(kMinVal, Math.round(invMin));
          const kHi = Math.min(kMaxVal, Math.round(invMax));
          const nextMin = Math.min(kLo, kHi);
          const nextMax = Math.max(kLo, kHi);
          setRangeMin(nextMin);
          setRangeMax(nextMax);
        });

      const gBrush = g.append("g").attr("class", "k-brush");
      gBrush.call(brush);
      gBrush.selectAll(".selection").attr("fill", "#d5bb9eff").attr("fill-opacity", 0.25);
      gBrush.selectAll(".handle").attr("fill", "#cfaa80ff");
      gBrush.call(brush.move, [x(effectiveMin), x(effectiveMax)]);
    }

    const paths = allDataPaths && allDataPaths.length ? allDataPaths : [dataPath];
    Promise.all(
      paths.map((p) =>
        fetch(p)
          .then((res) => {
            if (!res.ok) throw new Error("Failed to load AIC/BIC data");
            return res.json();
          })
          .catch((err) => {
            console.error(err);
            return null;
          }),
      ),
    )
      .then((list) => {
        const dataList = list.filter((d): d is AicBicData => !!d);
        allData = dataList;
        lastData = dataList.find((d) => d.model && d.layer && pMatch(d, dataPath)) ?? dataList[0] ?? null;
        const kValues = dataList[0]?.k_values ?? [];
        if (kValues.length) {
          const nextMin = Math.min(...kValues);
          const nextMax = Math.max(...kValues);
          setKMin(nextMin);
          setKMax(nextMax);
          setRangeMin((prev) => (prev < nextMin || prev > nextMax ? nextMin : prev));
          setRangeMax((prev) => (prev > nextMax || prev < nextMin ? nextMax : prev));
        }
        draw();
      })
      .catch((err) => {
        console.error(err);
        if (container) container.innerHTML = "<div style='color:red'>Failed to load AIC/BIC data.</div>";
      });

    function pMatch(d: AicBicData, path: string) {
      const expect = `/data/dimenet/bic_aic/${d.model}_${d.layer}.json`;
      return path === expect;
    }

    return () => {
      ro.disconnect();
    };
  }, [dataPath, allDataPaths, isSelected, activeMetric, rangeMin, rangeMax]);

  return (
    <div className="w-full h-full flex">
      <div ref={chartRef} className="h-full flex-1 min-w-0" />
      <div className="w-8 shrink-0 border-l border-zinc-200 flex flex-col items-start justify-start pt-2 pl-2 gap-2">
        <button
          type="button"
          className="text-[10px] font-semibold leading-none cursor-pointer"
          style={{ color: "#7A83BF", opacity: activeMetric === "bic" ? 0.45 : 1 }}
          onClick={() => setActiveMetric((prev) => (prev === "aic" ? "both" : "aic"))}
        >
          AIC
        </button>
        <button
          type="button"
          className="text-[10px] font-semibold leading-none cursor-pointer"
          style={{ color: "#B5BF4F", opacity: activeMetric === "aic" ? 0.45 : 1 }}
          onClick={() => setActiveMetric((prev) => (prev === "bic" ? "both" : "bic"))}
        >
          BIC
        </button>
      </div>
    </div>
  );
}
