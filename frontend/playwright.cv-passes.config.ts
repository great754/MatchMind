import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:['cv-passes.spec.ts','full-match.spec.ts','team-goal.spec.ts','player-focus.spec.ts','match-experience.spec.ts'],workers:1,timeout:240000,use:{baseURL:'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1440,height:1100},trace:'retain-on-failure'}});
