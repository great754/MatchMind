import {defineConfig} from '@playwright/test';
import base from './playwright.m6.config';
export default defineConfig({...base,testMatch:[...base.testMatch as string[],'camera-coverage.spec.ts']});
