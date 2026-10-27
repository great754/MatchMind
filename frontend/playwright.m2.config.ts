import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:['azure-clips.spec.ts','cv-review.spec.ts'],workers:1,timeout:360000,use:{baseURL:'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1440,height:1050},trace:'off'}});
