import {defineConfig} from '@playwright/test';
import base from './playwright.m2.config';
export default defineConfig({...base,testMatch:['azure-clips.spec.ts','temporal-world.spec.ts','team-goal.spec.ts','player-world.spec.ts','cv-review.spec.ts']});
