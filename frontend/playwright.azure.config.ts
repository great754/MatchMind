import {defineConfig} from '@playwright/test';
// Explicit live integration test: use already running Azure backend + Vite proxy.
export default defineConfig({testDir:'./tests',testMatch:'azure-clips.spec.ts',workers:1,timeout:300000,use:{baseURL:process.env.TACTIQAI_AZURE_FRONTEND_URL??'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1440,height:1050},trace:'retain-on-failure'}});
