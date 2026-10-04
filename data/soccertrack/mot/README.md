# SoccerTrack Challenge 2025 — MOT bounding-box ground truth

Downloaded 2026-08-29 from the organisers' public Google Drive
(`soccertrackchallenge@gmail.com`, folder `ScoccerChllenge2025`,
https://drive.google.com/drive/folders/1_o78gcL4j0xHxbRjSR1Evs4VLXCr2ncD).

After the challenge ended the organisers published the `.txt` ground truth for
**all 10 matches** (previously only training GT was distributed; test/challenge
GT lived on the CodaLab eval server). Layout on Drive:

- `Training Dataset/`: 117092, 117093, 118575, 118576, 118577, 128058
- `Challenge Videos/`: 128057, 132831
- `Test Videos/`: 118578, 132877

Each `.txt` pairs with a same-named `.mp4` clip in the same Drive folder —
annotations index frames of those clips, NOT the full-match panorama videos.
The clips are **4-minute** excerpts, 4096x1080 @ 25 fps (verified on
117093.mp4: 5,976 frames = exactly the GT frame count), every frame annotated
with all 22 players (boxes = frames x 22 exactly, all ten matches). The
annotators' source clips (fujii@nagoya Drive tree) were 10-minute cuts; the
released challenge clips are shorter.

Format: MOT — `frame,id,x,y,w,h,-1,-1,-1,-1`, pixel coords, top-left origin.

Per-file stats (rows / frame range / unique ids):

| match  | rows   | frames    | ids |
|--------|--------|-----------|-----|
| 117092 | 132000 | 0..5999   | 22  |
| 117093 | 131472 | 0..5975   | 22  |
| 118575 | 132000 | 0..5999   | 22  |
| 118576 | 131450 | 0..5974   | 22  |
| 118577 | 132000 | 0..5999   | 22  |
| 118578 | 132000 | 0..5999 (normalised, was 1..6000) | 22 |
| 128057 | 131472 | 0..5975   | 22  |
| 128058 | 131582 | 0..5980   | 22  |
| 132831 | 131472 | 0..5975   | 22  |
| 132877 | 131076 | 0..5957   | 22  |

Gotchas:
- `118578.txt` arrived 1-based from the organisers (the other nine 0-based).
  **Normalised to 0-based in this copy on 2026-08-31** (frames now 0..5999);
  the Drive original remains 1-based.
- Exactly 22 ids per match: outfield players + GKs only. No ball, no referees.
- Which 10-minute window of each match the clips cover, and clip fps/resolution,
  are not documented here — derive from the paired `.mp4`s if needed.

Drive file ids:
117092=1cQeC4c0FG8i8Xayw3LdzhEU5p42s7zD1  117093=1E34_ZbiIAC558niv_v0jHUxML6tfxBYu
118575=1Akbc1FIooHV3L9pKVq4KBMzzRURNt5AU  118576=1PFbhVcrwjtOZnD2CyKRwYGgrJU68Xbr1
118577=1cEu7_wymIitYV1tJXD1xoWz-UG_692qm  118578=1I9FxSKxBwM3pNBS6dGNUSmcemZsNSUoF
128057=1fbVEOEkW4aCyHbz8HtPp_GokYO0s3h6m  128058=1xUZNIwhGG5lVK-vwmn7a_T622gmGc6n1
132831=1DuuA1qGLeugqFIXlGfbxgcvd0DGQmsAj  132877=1FXRYMkgJ1xYwaSPzyAzPvzOCOGYZtDRJ
