import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:'m25-dataset.spec.ts',outputDir:'/tmp/tactiq-m25-browser',workers:1,timeout:90000,use:{baseURL:'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1440,height:1100}}});
