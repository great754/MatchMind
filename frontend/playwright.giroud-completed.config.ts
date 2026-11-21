import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:'giroud-completed.spec.ts',outputDir:'/tmp/tactiq-giroud-completed-browser-artifacts',workers:1,timeout:180000,
 expect:{timeout:30000},use:{baseURL:'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1440,height:1100},trace:'off'}});
