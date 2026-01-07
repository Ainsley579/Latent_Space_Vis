import { useEffect, useState } from "react";
import SankeyDiagram from "./components/SankeyDiagram";
import ScatterPlot from "./components/ScatterPlot";
import WindowManager from "./components/WindowManager";

type AppProps = {
    initialData?: any[];
    dataUrl?: string;
};


export function Home() {
    return (
        <main className="min-h-screen bg-gray-100">
            <WindowManager />
        </main>
    );
}

export function SankeyApp({ initialData, dataUrl }: AppProps) {
    return (
        <div style={{ padding: 8 }}>
            <h3>Sankey (Notebook)</h3>
            <SankeyDiagram data={initialData} zoomFactor={1} />
        </div>
    );
}

export function ScatterApp({ initialData, dataUrl }: AppProps) {
    return (
        <div style={{ padding: 8 }}>
            <h3>Scatter (Notebook)</h3>
            <ScatterPlot data={initialData} zoomFactor={1} />
        </div>
    );
}
