import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  define: {
    // polyfill minimal process env so the bundle doesn't throw "process is not defined"
    'process.env': {},
  },
  build: {
    lib: {
      entry: "src/app.tsx",
      name: "NotebookVis",
      formats: ["iife"],   // 关键：IIFE，才能内联执行
      fileName: () => "notebook-vis.js",
    },
    rollupOptions: {
      external: [],       // 不 externalize react
    },
  },
});