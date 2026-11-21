import {afterEach,describe,it,expect,vi} from 'vitest';
import {cleanup,render,screen} from '@testing-library/react';
import {intelligentFrameSchema} from './intelligentContracts';
import IntelligentTracking,{IntelligentOverlay,CompletedSourceReview} from './IntelligentTracking';
import type {BallState} from './BallDebug';
afterEach(()=>{cleanup();vi.restoreAllMocks()});
const sample=()=>({schema_version:'cv-m23-frame-v1',recording_id:'synthetic',variant:'full_m23',frame:3,timestamp:.12,camera_epoch:0,state:'OBSERVED',directly_observed:true,selected_candidate_id:'current',
 candidates:[{candidate_id:'current',bbox:[20,30,5,5],confidence:.7,origin:'guided',visual_evidence:'CURRENT_PIXELS',observation_eligible:true,selected:true,decision_reason:'CURRENT_VISUAL_EVIDENCE',blur_supported:false}],memory:{state:'PREDICTED',missed_frames:1,missed_seconds:.04,last_direct_observation:{position:[20,30],frame:2,timestamp:.08},hypotheses:[{mean:[20,30],covariance:[[9,0],[0,16]],velocity:[1,2],acceleration:[0,0],track_id:1,confirmed:true,archived:false,age_seconds:.04}],search_regions:[{bbox:[0,0,80,80],track_id:1,kind:'trajectory'}],camera_motion_source:'measured',camera_cut:false,reacquired_with_current_visual_evidence:true,predicted_image_position:null,reason:'FRESH_PIXELS'},reconstruction:null,trajectory_history:[{frame:2,current_camera_image:[20,30],current_observation:false}],interaction:{state:'UNKNOWN',reason:'NO_CONTACT',verified_touch:false,verified_possession:false},reasons:['CURRENT_VISUAL_SELECTION'],source_pixels_sha256:'a'.repeat(64),model_set_sha256:'b'.repeat(64),config_sha256:'c'.repeat(64),uses_future_for_association:true,future_visual_frames_used:false});
describe('M23 visual evidence contract',()=>{
 const completed=()=>({completed_tasks:40,visible_frames:31,uncertain_frames:9,absent_frames:0,rejected_regions:49,
  systems:[{id:'full_m23',ball:{tp:1,fp:6,fn:30,precision:1/7,recall:1/31,f1:1/19}}],failure_counts:{no_matching_raw_candidate:12},
  evaluated_at_utc:'2026-11-21T00:00:00Z',annotation_snapshot_sha256:'d'.repeat(64),labels_updated_since_snapshot:false,
  reference_role:'development_only' as const,model_assisted:true as const,independent_final_test:false as const});
 it('shows completed review metrics with the actual known-visibility denominator',()=>{
  render(<CompletedSourceReview review={completed()}/>);
  expect(screen.getByText(/31 visible balls · 9 uncertain frames excluded/)).toBeTruthy();
  const table=screen.getByRole('table',{name:'Completed ball review comparison'});
  expect(table).toHaveTextContent('1 / 6 / 30');expect(table).toHaveTextContent('14.3%');expect(table).toHaveTextContent('3.2%');expect(table).toHaveTextContent('0.053');
  expect(screen.getByText(/do not establish independent final-test accuracy/)).toBeTruthy();
 });
 it('marks an evaluation snapshot stale after a later label correction',()=>{
  render(<CompletedSourceReview review={{...completed(),labels_updated_since_snapshot:true}}/>);
  expect(screen.getByRole('status')).toHaveTextContent('labels changed after this evaluation snapshot');
 });
 it('replaces the unlabelled warning with the validated completed review',async()=>{
  const meta={recording_id:'giroud',reference_role:'reviewed_development',model_set_sha256:'b'.repeat(64),ball_revision:'c'.repeat(64),
   promotion:{status:'not_promoted',ordinary_default:'m14',reason:'Development evidence'},independent_final_test:'unavailable',
   completed_review:completed(),metrics:{navigation:[],engineering:{states:{OBSERVED:67}},pass_traces:[]},comparisons:[],frozen_baselines:[],
   runtime:{processing_fps:2,recorded_processing_seconds:3.5,timing:{extra_crop_views:4},peak_rss_bytes_macos:1024}};
  vi.spyOn(globalThis,'fetch').mockResolvedValue(new Response(JSON.stringify(meta),{status:200}));
  render(<IntelligentTracking recordingId="giroud" video={null} width={640} height={360} fps={25} onSeek={()=>{}}/>);
  expect(await screen.findByRole('table',{name:'Completed ball review comparison'})).toBeTruthy();
  expect(screen.queryByTestId('m23-unlabelled')).toBeNull();expect(screen.queryByLabelText('M23 evaluation sample')).toBeNull();
 });
 it('shows unlabelled source coverage without invented accuracy tables',async()=>{
  const meta={recording_id:'giroud',reference_role:'unlabelled_inference',model_set_sha256:'b'.repeat(64),ball_revision:'c'.repeat(64),
   promotion:{status:'not_promoted',ordinary_default:'m14',reason:'No independent references'},independent_final_test:'unavailable',
   metrics:{navigation:[],engineering:{states:{OBSERVED:5,UNKNOWN:2}},pass_traces:[]},comparisons:[],frozen_baselines:[],
   runtime:{processing_fps:2,recorded_processing_seconds:3.5,timing:{extra_crop_views:4},peak_rss_bytes_macos:1024}};
  vi.spyOn(globalThis,'fetch').mockResolvedValue(new Response(JSON.stringify(meta),{status:200}));
  render(<IntelligentTracking recordingId="giroud" video={null} width={640} height={360} fps={25} onSeek={()=>{}}/>);
  expect(await screen.findByTestId('m23-unlabelled')).toHaveTextContent('have not been measured');
  expect(screen.queryByRole('table',{name:'M23 controlled comparisons'})).toBeNull();
  expect(screen.queryByLabelText('M23 evaluation sample')).toBeNull();
  expect(screen.getByText(/Whole clip: OBSERVED: 5/)).toBeTruthy();
 });
 it('accepts verified source pixels with separate causal memory',()=>{expect(intelligentFrameSchema.parse(sample()).state).toBe('OBSERVED')});
 it('rejects predicted data masquerading as observed',()=>{expect(intelligentFrameSchema.safeParse({...sample(),state:'PREDICTED'}).success).toBe(false)});
 it('rejects a selected below-threshold candidate',()=>{const s=sample();s.candidates[0].confidence=.2;expect(intelligentFrameSchema.safeParse(s).success).toBe(false)});
 it('requires current-frame M22 confidence as well as temporal evidence',()=>{const s=sample();s.candidates[0].origin='complement';expect(intelligentFrameSchema.safeParse(s).success).toBe(false)});
 it('rejects invalid covariance and nonfinite coordinates',()=>{const s=sample();s.memory.hypotheses[0].covariance=[[1,2],[2,1]];expect(intelligentFrameSchema.safeParse(s).success).toBe(false);s.candidates[0].bbox[0]=Infinity;expect(intelligentFrameSchema.safeParse(s).success).toBe(false)});
 it('renders search, uncertainty and real observations on their exact frame',()=>{const frame=intelligentFrameSchema.parse(sample());render(<svg><IntelligentOverlay frame={frame} ball={{frame:3} as BallState}/></svg>);expect(screen.getByTestId('m23-search')).toBeTruthy();expect(screen.getByTestId('m23-uncertainty')).toBeTruthy();expect(screen.getByTestId('m23-observed')).toBeTruthy();expect(screen.queryByTestId('m23-reconstructed')).toBeNull()});
 it('hides stale overlays when the decoded source frame differs',()=>{const frame=intelligentFrameSchema.parse(sample());render(<svg><IntelligentOverlay frame={frame} ball={{frame:4} as BallState}/></svg>);expect(screen.queryByTestId('m23-overlay')).toBeNull()});
 it('does not show unselected candidates by default',()=>{const s=sample();s.candidates.push({...s.candidates[0],candidate_id:'other',selected:false});const frame=intelligentFrameSchema.parse(s);render(<svg><IntelligentOverlay frame={frame} ball={{frame:3} as BallState}/></svg>);expect(screen.queryByTestId('m23-rejected')).toBeNull()});
});
