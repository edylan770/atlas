import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
        // Large chunked ingest uploads (e.g. 25 images) can take several minutes.
        timeout: 600_000,
        proxyTimeout: 600_000,
      },
    },
  },
});
