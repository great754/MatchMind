import {defineConfig} from '@playwright/test';
import base from './playwright.m16.config';
export default defineConfig({...base,testMatch:[...base.testMatch as string[],'highlights.spec.ts']});
