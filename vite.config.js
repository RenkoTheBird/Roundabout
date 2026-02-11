import { defineConfig } from 'vite'
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
      
      function copyRecursive(src, dest) {
        if (!existsSync(dest)) {
          mkdirSync(dest, { recursive: true });
        }
        
        const entries = readdirSync(src, { withFileTypes: true });
        
        for (const entry of entries) {
          const srcPath = join(src, entry.name);
          const destPath = join(dest, entry.name);
          
          // Skip popup directory as it's handled by Vite build
          if (entry.name === 'popup') continue;
          
          if (entry.isDirectory()) {
            copyRecursive(srcPath, destPath);
          } else {
            copyFileSync(srcPath, destPath);
          }
        }
      }
      
      // Copy all files except popup (which is built by Vite)
      const entries = readdirSync(publicDir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.name === 'popup') continue;
        
        const srcPath = join(publicDir, entry.name);
        const destPath = join(distDir, entry.name);
        
        if (entry.isDirectory()) {
          copyRecursive(srcPath, destPath);
        } else {
          copyFileSync(srcPath, destPath);
        }
      }
    }
  };
}

// https://vite.dev/config/
export default defineConfig({
  root: "src/public",
  plugins: [react(), copyPublicFiles()],
  build: {
    rollupOptions: {
      input: {
        popup: resolve(__dirname, "src/public/popup/index.html"),
      },
    },
    outDir: "../../dist",
    emptyOutDir: true,
  },
});
