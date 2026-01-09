import React from "react";
import { createRoot } from "react-dom/client";
import { SankeyApp, ScatterApp, Home } from "./app";

export function SankeyDiagram(id, data) {
    const root = createRoot(document.querySelector(id));
    const el = React.createElement(SankeyApp, { initialData: data, dataUrl: "" })
    root.render(el);
}

export function Scatterplot(id, data) {
    const root = createRoot(document.querySelector(id));
    const el = React.createElement(ScatterApp, {
        initialData: data.data,
        images: data.images,
        layers: data.layers,
        dataUrl: ""
    })
    root.render(el);
}

export default function App(div) {
    const root = createRoot(div);
    const el = React.createElement(Home)
    root.render(el);
}
