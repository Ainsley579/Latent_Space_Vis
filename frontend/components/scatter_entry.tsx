import React from "react";
import { createRoot } from "react-dom/client";
import ScatterPlot from "./ScatterPlot";

let root: any = null;

/**
 * Render ScatterPlot into a given container.
 * This is the React → DOM adapter layer.
 */
export function renderScatterPlot(
  container: HTMLElement,
  props: any
) {
  if (!root) {
    root = createRoot(container);
  }

  root.render(<ScatterPlot {...props} />);
}