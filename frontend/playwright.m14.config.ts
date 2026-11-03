import {defineConfig} from '@playwright/test';
import base from './playwright.m9.config';
export default defineConfig({...base,testMatch:[...base.testMatch as string[],'ball-world.spec.ts']});
