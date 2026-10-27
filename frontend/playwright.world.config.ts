import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:'world-model.spec.ts',workers:1,timeout:90000,use:{baseURL:'http://127.0.0.1:18765',channel:'chrome',viewport:{width:1500,height:1000},trace:'retain-on-failure'}});
