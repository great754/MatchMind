import {defineConfig} from '@playwright/test';
import base from './playwright.m17.config';
export default defineConfig({...base,globalSetup:'./tests/helpers/ready-m18.ts',testMatch:[...base.testMatch as string[],'unified-assistant.spec.ts'],timeout:180000});
