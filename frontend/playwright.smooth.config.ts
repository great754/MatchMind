import {defineConfig} from '@playwright/test';
import base from './playwright.m7.config';
export default defineConfig({...base,testMatch:['cv-smooth.spec.ts','cv-review.spec.ts','team-goal.spec.ts','player-world.spec.ts','camera-coverage.spec.ts','temporal-world.spec.ts']});
