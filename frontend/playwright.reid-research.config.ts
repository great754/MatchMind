import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:['soccer-reid-research.spec.ts'],outputDir:'test-results/reid',workers:1,timeout:60000,use:{baseURL:'http://127.0.0.1:5174',channel:'chrome',viewport:{width:1440,height:1000},trace:'off'}});
