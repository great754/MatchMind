import {defineConfig} from '@playwright/test';
import base from './playwright.m7.config';
export default defineConfig({...base,testMatch:[...base.testMatch as string[],'southampton.spec.ts','cv-smooth.spec.ts']});
