"""Render match footage with a synchronized, team-colored tactical inset."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.vision.coordinate_transform import PitchCoordinateTransformer
from src.vision.data_loading import DatasetPaths, load_mot
from src.vision.pitch import PITCH_LENGTH, PITCH_WIDTH, create_pitch, draw_player
from src.vision.annotations import ClipAnnotations
from src.vision.tactical_tracking import prepare_tactical_tracking
from src.vision.video_renderer import MatchMindVideoWriter, overlay_tactical_view


def valid_pitch_position(x, y):
    return bool(np.isfinite([x, y]).all() and 0 <= x <= PITCH_LENGTH and 0 <= y <= PITCH_WIDTH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/match_118575_matchmind.mp4')
    parser.add_argument('--max-frames', type=int, help='Render only this many frames for a preview')
    parser.add_argument('--inset-width', type=float, default=0.215)
    parser.add_argument('--opacity', type=float, default=0.65)
    parser.add_argument('--annotations', action='store_true', help='Render synchronized 12-class action annotations')
    parser.add_argument('--play', action='store_true')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if not 0.05 <= args.inset_width <= 0.35 or not 0 <= args.opacity <= 1:
        parser.error('inset width must be 0.05–0.35 and opacity must be 0–1')

    paths = DatasetPaths()
    mot = load_mot(paths.mot)
    cap = cv2.VideoCapture(str(paths.video))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {paths.video}')
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or mot.frame.min() != 0 or mot.frame.max() != total - 1 or mot.frame.nunique() != total:
            raise ValueError('Video and zero-based tracking frame counts do not match')
        print(f'Video: {width}x{height}, {fps:g} FPS, {total} frames', flush=True)
        print('Comparing GSR and calibrated image positions...', flush=True)
        tracking, frame_errors = prepare_tactical_tracking(paths, mot, fps)
        mot, teams = tracking.rows, tracking.teams
        comparison = tracking.comparison
        print(f"Comparison across {comparison['frames_compared']} frames: "
              f"mean {comparison['mean_m']} m, median {comparison['median_m']} m, "
              f"P95 {comparison['p95_m']} m (relative to GSR)", flush=True)
        print(f"Selected sources: {comparison['position_source_counts']}", flush=True)
        print(f"Goalkeepers: {comparison['goalkeepers']}", flush=True)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix('.comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n')
        frame_errors.to_csv(output.with_suffix('.comparison.csv'), index=False)
        output.with_suffix('.positions.csv').write_text(mot[['frame','player_id','pitch_x','pitch_y','position_source','is_goalkeeper','transform_gsr_error_m']].to_csv(index=False))
        output.with_suffix('.teams.json').write_text(json.dumps(teams, indent=2)+'\n')
        if tracking.sync:
            output.with_suffix('.sync.json').write_text(json.dumps(tracking.sync, indent=2)+'\n')
        transformer = PitchCoordinateTransformer(paths.keypoints)
        print(f'Landmark calibration: {transformer.validation_error()}', flush=True)
        annotations = ClipAnnotations(paths, transformer, total, fps) if args.annotations else None
        if annotations:
            output.with_suffix('.events.json').write_text(json.dumps(annotations.events, indent=2)+'\n')
            output.with_suffix('.sync.json').write_text(json.dumps(annotations.sync, indent=2)+'\n')
            print(f'Aligned {len(annotations.events)} actions; ball offset {annotations.offset}', flush=True)
        by_frame = {int(n): rows for n, rows in mot.groupby('frame')}
        base_pitch = create_pitch()
        writer = MatchMindVideoWriter(str(output), width, height, fps)
        limit = min(total, args.max_frames) if args.max_frames else total
        rendered = 0
        for frame_number in range(limit):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Video ended unexpectedly at frame {frame_number}')
            tactical = base_pitch.copy()
            for player in by_frame[frame_number].itertuples():
                if valid_pitch_position(player.pitch_x, player.pitch_y):
                    draw_player(tactical, player.pitch_x, player.pitch_y,
                                int(player.player_id), team=teams[int(player.player_id)], show_id=False, is_goalkeeper=bool(player.is_goalkeeper))
            combined = overlay_tactical_view(frame, tactical, width_ratio=args.inset_width,
                                              bottom_margin=18, opacity=args.opacity)
            cv2.putText(combined, f'{frame_number / fps:.2f}s', (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            for label, color, y in [('BLUE TEAM', (255, 80, 80), 72), ('RED TEAM', (80, 80, 255), 108)]:
                cv2.circle(combined, (28, y - 8), 8, color, -1, cv2.LINE_AA)
                cv2.putText(combined, label, (46, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2, cv2.LINE_AA)
            if annotations:
                actor = annotations.draw(combined, frame_number)
                if actor is not None:
                    rows = by_frame[frame_number]
                    selected = rows[rows.player_id == actor]
                    if not selected.empty:
                        p = selected.iloc[0]
                        cv2.ellipse(combined, (int(p.foot_x), int(p.foot_y)),
                                    (18, 7), 0, 0, 360, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(combined)
            if frame_number == 0:
                cv2.imwrite(str(output.with_suffix('.jpg')), combined)
            rendered += 1
            if rendered % 250 == 0:
                print(f'Rendering {rendered}/{limit}', flush=True)
            if args.play:
                cv2.imshow('MatchMind', cv2.resize(combined, (1920, int(height * 1920 / width))))
                if cv2.waitKey(max(1, round(1000 / fps))) & 0xFF == ord('q'):
                    break
        print(f'Saved {rendered} frames to {output}', flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.close()
        if args.play:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
