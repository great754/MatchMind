import {describe,it,expect} from 'vitest';
import {render,screen,cleanup} from '@testing-library/react';
import {CameraDebug,CameraProjectionDetails} from './CameraDebug';
const motion=(status:string)=>({localization:{status,outcome:status==='UNKNOWN'?'PROPAGATION_EXPIRED':'TEMPORALLY_TRACKED',absolute_outcome:'POSE_RESIDUAL_TOO_HIGH',tracking_blockers:status==='UNKNOWN'?['PROPAGATION_EXPIRED']:[],semantic_classes:['Side line top','Middle line'],transition:status==='RELOCALIZED'?'relocalization':null,quality:'measured motion; increased uncertainty',last_absolute_age_seconds:.125,camera_epoch:2},projective_motion:{matches:80,inlier_ratio:.95,heldout_median_small_px:.2,heldout_p90_small_px:.7}});
describe('camera diagnostics',()=>{
 it.each(['DIRECT','TRACKED','TEMPORAL_TRACKED','PARTIAL_CONSTRAINED','RELOCALIZED','UNKNOWN'])('shows supported %s provenance and measured checks',status=>{
  cleanup();render(<CameraDebug motion={motion(status)}/>);
  expect(screen.getByTestId('cv-camera-debug').getAttribute('data-status')).toBe(status);
  expect(screen.getByText(/Last absolute calibration/).textContent).toContain('0.125s ago');
  expect(screen.getByText(/Pitch evidence/).textContent).toContain('2 semantic classes');
  if(status==='UNKNOWN')expect(screen.getByText(/Localization unavailable/).textContent).toContain('PROPAGATION_EXPIRED');
  cleanup();
 });
 it('renders M5 field motion in its actual working coordinates',()=>{const m=motion('PARTIAL_CONSTRAINED');render(<CameraDebug motion={{...m,field_motion:{matches:44,inlier_ratio:1,heldout_p90_small_px:.2,scale_to_source:1.6875}}}/>);expect(screen.getByText(/Motion matches/).textContent).toContain('640px working width; source scale 1.688');cleanup()});
 it('withholds unsupported diagnostic values',()=>{render(<CameraDebug motion={{localization:{status:'CONFIDENT'}}}/>);expect(screen.getByTestId('cv-camera-debug').getAttribute('data-status')).toBe('UNKNOWN');cleanup()});
});

it('shows readable per-player camera and source foot provenance',()=>{render(<CameraProjectionDetails evidence={{state:'PARTIAL_CONSTRAINED',source:'partial_constrained',epoch:2,support:.812345,last_direct_age_seconds:1.125,current_semantic_support:true,image_foot:[150.1234,700.5678]}}/>);expect(screen.getByTestId('player-camera-evidence').textContent).toContain('150.12, 700.57 pixels');expect(screen.getByTestId('player-camera-evidence').textContent).toContain('0.812 (heuristic)');cleanup()});
it.each(['TEMPORAL_PREDICTED','TEMPORAL_CORRECTED','BIDIRECTIONAL_INTERPOLATED'])('renders M6 %s state without collapsing evidence',status=>{render(<CameraDebug motion={motion(status)}/>);expect(screen.getByTestId('cv-camera-debug').getAttribute('data-status')).toBe(status);cleanup()});
