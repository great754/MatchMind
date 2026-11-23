import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:'m24-benchmark.spec.ts',outputDir:'/tmp/tactiq-m24-browser-artifacts',workers:1,timeout:180000,expect:{timeout:30000},use:{baseURL:'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1500,height:1100},trace:'retain-on-failure'}});
