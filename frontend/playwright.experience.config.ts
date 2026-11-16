import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:['player-focus.spec.ts','match-experience.spec.ts','full-match.spec.ts','team-goal.spec.ts'],workers:1,timeout:240000,use:{baseURL:process.env.TACTIQAI_EXPERIENCE_URL??'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1440,height:1100},trace:'retain-on-failure'}});
