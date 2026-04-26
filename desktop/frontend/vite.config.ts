import { defineConfig } from "vite";

// Tauri talks to the dev server on port 5173 (matched in tauri.conf.json).
// Disable HMR overlay for now — Phase 2 will integrate it with the
// Tauri devtools properly. Strict port so the backend's hardcoded URL
// fails loudly instead of silently spawning a different port.
export default defineConfig({
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "esnext",
  },
});
