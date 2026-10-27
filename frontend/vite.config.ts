import { defineConfig } from 'vite';
import {dirname,join} from 'node:path';
import {realpathSync} from 'node:fs';
import react from '@vitejs/plugin-react';
// Same-origin dev proxy. No provider keys/env files are read or bundled here.
export default defineConfig({
 plugins:[react()],
 cacheDir:process.env.TACTIQAI_VITE_CACHE_DIR??join(dirname(realpathSync('node_modules')),'vite-cache'),
 optimizeDeps:{entries:['index.html'],include:['react','react-dom/client','react/jsx-runtime','react/jsx-dev-runtime','lucide-react','zod']},
 server:{
  fs:{allow:['.','../src/serving/static']},
  host:'127.0.0.1',port:Number(process.env.TACTIQAI_DEV_PORT??5173),strictPort:true,
  // File-provider hydration emits native change events without source edits.
  // Polling compares actual file timestamps instead and keeps HMR working.
  watch:{usePolling:true,interval:500,ignored:['**/node_modules.cloud-backup*/**','**/test-results/**','**/playwright-report/**']},
  proxy:{'/matches':{target:'http://127.0.0.1:'+String(process.env.TACTIQAI_DEV_API_PORT??8000),changeOrigin:false},'/api':{target:'http://127.0.0.1:'+String(process.env.TACTIQAI_DEV_API_PORT??8000),changeOrigin:false,rewrite:path=>path.replace(/^\/api/,'')}}
 }
});
