import {defineConfig} from '@playwright/test';
import base from './playwright.m9.config';
export default defineConfig({...base,testMatch:[...base.testMatch as string[],'ball-sixteen.spec.ts','ball-world.spec.ts','ball-motion.spec.ts','playback-seek.spec.ts','demo.spec.ts','second-recording.spec.ts']});
