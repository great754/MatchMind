import {defineConfig} from '@playwright/test';
import base from './playwright.m8.config';
export default defineConfig({...base,testMatch:[...base.testMatch as string[],'player-intelligence.spec.ts']});
