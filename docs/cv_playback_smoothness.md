# CV playback smoothness — verified 2026-10-27

**IMPLEMENTED / LOCALLY VERIFIED.** Fixed a frontend display bottleneck: CVReview previously requested/displayed observations only after the browser's low-frequency `timeupdate` event. The user reported roughly two updates per second; no numerical baseline benchmark was recorded.

The view now follows decoded video frames with `requestVideoFrameCallback`, with requestAnimationFrame fallback. A bounded buffer prefetches1.5seconds of existing observations with at most four concurrent requests and120 cached rows. Shared requests are not aborted every playback tick; obsolete responses never replace the current frame, and unmount aborts recording-specific work. Missing data stays missing. Static camera-timeline bars are memoized to avoid rebuilding541/500 elements every frame.

Decoded rational-FPS timestamps are snapped to their actual frame boundary to handle microsecond rounding. Paused diagnostic seeks retain the requested media time and preceding-frame semantics. No coordinate interpolation, trajectory filtering, detector change, new model run or Azure modification was made.

## Actual Chrome measurements

| Recording | Source FPS | Measured overlay updates/sec | Missing overlay samples | Maximum observed lag | Maximum gap |
|---|---:|---:|---:|---:|---:|
| test_match |23.976|**23.943**|0%|1frame|66.2ms|
| team_goal |25|**24.764**|0%|1frame|52.2ms|

Each measurement covered about2.5seconds of ordinary playback after a one-second warmup, with default review overlays. This is a local Chrome sample, not a guarantee for every device, debug combination or network. Underlying model/identity/camera jumps and unsupported intervals remain; this improves presentation cadence rather than geometric accuracy.

**79 frontend tests across13files passed;TypeScript/production build passed;all six affected CV Chrome suites passed in29.2seconds.** Regression covers both videos, playback, exact seeks, stepping, unknown coordinates, camera/team/player evidence, stale request isolation and recording changes. The new browser test measures actual DOM observation updates during playback. Backend and Azure code/data were unchanged; M7 camera coverage and earlier live Azure verification remain unchanged.

Changed:`frontend/src/videoFrames.ts`,its unit tests,`CVReview.tsx`,`CameraTimeline.tsx`,new`cv-smooth.spec.ts`and`playwright.smooth.config.ts`. Proof:`outputs/world_model/playback_smoothness_verification/`.

Refresh **http://127.0.0.1:5173**. No reprocessing or configuration change is needed. Normal Chrome test command from frontend with Node22:

```sh
npm test
npm run build
./node_modules/.bin/playwright test --config playwright.smooth.config.ts
```
