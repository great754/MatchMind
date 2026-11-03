import {defineConfig} from '@playwright/test';
import base from './playwright.southampton.config';
export default defineConfig({...base,testMatch:[...base.testMatch as string[],'metric-camera.spec.ts']});
