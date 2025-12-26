import { renderScatterPlot } from "./scatter_entry.tsx";

declare global {
  interface Window {
    latentvis: any;
  }
}

window.latentvis = {
  main: function (data: any, callbacks: any = {}) {
    let container = document.getElementById("latentvis-scatter");
    if (!container) {
      container = document.createElement("div");
      container.id = "latentvis-scatter";
      container.style.width = "100%";
      container.style.height = "300px";
      document.body.appendChild(container);
    }

    const scatterData = data.map((d: any) => ({
        x: d.x,
        y: d.y,
        real_id: d.id,
        layer: d.layer,
        color_value: d.color_value,
    }));

    renderScatterPlot(container, {
      data: scatterData,
      hideControls: true,
      onSelect: callbacks?.onSelect,
    });
  }
};