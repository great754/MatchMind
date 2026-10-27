import {defineConfig} from '@playwright/test';
import base from './playwright.m2.config';
export default defineConfig({...base,testMatch:['team-goal.spec.ts']});
