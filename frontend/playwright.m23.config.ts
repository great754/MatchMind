import {defineConfig} from '@playwright/test';
// Each project gets a fresh browser worker. Long video/renderer lifetimes from
// earlier suites must not carry into another pipeline's playback checks.
const suites=['m23','m22','m21','m20','m19','cv-passes','full-match','team-goal','player-focus','match-experience'];
export default defineConfig({testDir:'./tests',workers:1,timeout:600000,expect:{timeout:30000},projects:suites.map(name=>({name,testMatch:`${name}.spec.ts`})),use:{baseURL:process.env.TACTIQAI_BROWSER_URL??'http://127.0.0.1:5173',channel:'chrome',viewport:{width:1440,height:1100},trace:process.env.TACTIQAI_BROWSER_TRACE==='1'?'retain-on-failure':'off'}});
