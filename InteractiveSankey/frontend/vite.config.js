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
            entry: "src/main.js",
            name: "latent_space",
            // formats: ["iife"],   
            fileName: () => "bundle.js",
        },
        rollupOptions: {
            external: [],
        },
    },
});
