# CourtVision architectural attribution for M22

TactiqAI's M22 implementation adapts the three-frame ResNet-18/FPN/CenterNet design and visibility-aware training methodology studied in [CourtVision](https://github.com/aiden-guan/courtvision), commit `581f67e82c9e271751baffb62046cee4e1e94902`. Football source mapping, source-fold training, evidence validation and application integration are implemented in this repository. No basketball checkpoints, pseudo-label collection, event rules or footage are redistributed.

The upstream code is MIT-licensed. Preserve its notice with architectural adaptations; model weights and data have separate terms.

```text
MIT License

Copyright (c) 2026 Aiden Guan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
