import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path';
import { copyFileSync, mkdirSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';

// Plugin to copy static files from src/public to dist root
function copyPublicFiles() {
  return {
    name: 'copy-public-files',
    writeBundle() {
      const publicDir = resolve(__dirname, 'src/public');
      const distDir = resolve(__dirname, 'dist');

      function copyRecursive(src, dest, skipFileName) {
        if (!existsSync(dest)) {
          mkdirSync(dest, { recursive: true });
        }

        const entries = readdirSync(src, { withFileTypes: true });

        for (const entry of entries) {
          const srcPath = join(src, entry.name);
          const destPath = join(dest, entry.name);

          if (entry.name === 'popup') continue;

          if (entry.isDirectory()) {
            copyRecursive(srcPath, destPath, skipFileName);
          } else if (skipFileName && entry.name === skipFileName) {
            // Skip so Vite-built file is not overwritten
            continue;
          } else {
            copyFileSync(srcPath, destPath);
          }
        }
      }

      const entries = readdirSync(publicDir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.name === 'popup') continue;

        const srcPath = join(publicDir, entry.name);
        const destPath = join(distDir, entry.name);

        if (entry.isDirectory()) {
          const skipFile = entry.name === 'background' ? 'background.js' : null;
          copyRecursive(srcPath, destPath, skipFile);
        } else {
          copyFileSync(srcPath, destPath);
        }
      }
    }
  };
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const envDir = resolve(__dirname, "src/public");
  const env = loadEnv(mode, envDir, "");
  const braveKey = env.VITE_BRAVE_API_KEY || "";
  return {
  root: "src/public",
  define: {
    __BRAVE_API_KEY__: JSON.stringify(braveKey),
  },
  plugins: [react(), copyPublicFiles()],
  build: {
    rollupOptions: {
      input: {
        popup: resolve(__dirname, "src/public/popup/index.html"),
        background: resolve(__dirname, "src/public/background/background.js"),
      },
      output: {
        entryFileNames: (chunk) =>
          chunk.name === "background" ? "background/background.js" : "[name].js",
        manualChunks: (id) => {
          if (id.includes("node_modules/@xenova") || id.includes("node_modules/onnxruntime")) {
            return "transformers";
          }
        },
      },
      onwarn(warning, warn) {
        if (warning.code === "EVAL" && warning.id?.includes("onnxruntime")) return;
        warn(warning);
      },
    },
    outDir: "../../dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
};
});
