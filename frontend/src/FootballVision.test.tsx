import {render,screen,cleanup,within} from '@testing-library/react';
import {afterEach,it,expect,vi} from 'vitest';
import FootballVision from './FootballVision';
import {request} from './api';
vi.mock('./api',()=>({request:vi.fn()}));
afterEach(()=>{cleanup();vi.clearAllMocks()});
const counts={tp:1,fp:1,fn:1,precision:.5,recall:.5,f1:.5};
const metrics={ball:counts,passes:{...counts,tp:0,fp:0,fn:2,precision:null,recall:0,f1:0}};
const metadata={recording_id:'test_match',reference_role:'development_only',checkpoint_sha256:'a'.repeat(64),
 model:{architecture:'three-frame R18',epoch:1,split:{train_sources:['team_goal','southampton'],validation_sources:['team_goal','southampton'],excluded_development_source:'test_match'}},
 metrics,aggregate:metrics,baselines:{m20_control:{aggregate:metrics,per_recording:{test_match:metrics}},offline_dag:{aggregate:metrics,per_recording:{test_match:metrics}}},
 promotion:{status:'not_promoted',reason:'Development only'},independent_test_status:'not_available',training:{verified_training_frames:85,verified_validation_frames:25,epochs_run:8,weak_training_frames:120},runtime:{processing_fps:1,inference_seconds:541,peak_rss_bytes_macos:1000000},pass_trace:[]};
function show(value:object){vi.mocked(request).mockResolvedValue(value);render(<FootballVision recordingId="test_match" frame={null} ball={null} m20={null} video={null} width={1080} height={1080} fps={25} onSeek={()=>{}}/>)}
it('keeps the original development comparison available for older artifacts without completed-review metrics',async()=>{
 show(metadata);await screen.findByRole('heading',{name:'M22 — Temporal Football Detector'});
 expect(screen.getByRole('table',{name:'M20 M21 M22 development accuracy'})).toBeInTheDocument();expect(screen.queryByText('Results from your completed M22 review')).toBeNull();
});
it('separates newly reviewed-frame metrics from original references and never calls them a final test',async()=>{
 show({...metadata,completed_review:{frames:150,visible_frames:145,systems:[{id:'m22_reviewed',ball:{...counts,tp:40,fp:2,fn:10}}],combined:{...counts,tp:48,fp:4,fn:13}}});
 await screen.findByText('Results from your completed M22 review');const table=screen.getByRole('table',{name:'Completed M22 review comparison'});
 expect(within(table).getByText('Retrained M22')).toBeInTheDocument();expect(table).toHaveTextContent('40 / 2 / 10');expect(screen.getByText(/not an independent final test/)).toBeInTheDocument();
 expect(screen.getByText(/Combining original and new known ball frames/)).toHaveTextContent('48 correct, 4 false, 13 missed');
 expect(screen.getByText(/Pass detection is still evaluated against the original seven reference passes/)).toBeInTheDocument();
});
