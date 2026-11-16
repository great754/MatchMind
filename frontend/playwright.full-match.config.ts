import {defineConfig} from '@playwright/test';
export default defineConfig({globalSetup:'./tests/helpers/ready-m18.ts',testDir:'./tests',testMatch:['full-match.spec.ts','team-goal.spec.ts'],workers:1,timeout:240000,use:{baseURL:'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1440,height:1050},trace:'retain-on-failure'}});
