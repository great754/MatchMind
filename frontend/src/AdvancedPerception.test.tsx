import {render,screen,fireEvent,cleanup} from '@testing-library/react';
import {afterEach,describe,it,expect,vi} from 'vitest';
import AdvancedPerception,{overlayMatchesFrame} from './AdvancedPerception';
import {advancedMetadataSchema,experimentLabel} from './advancedContracts';
afterEach(cleanup);
const metric={ball:{tp:0,fp:1,fn:11,precision:0,recall:0,f1:0},passes:{tp:0,fp:0,fn:3,precision:null,recall:0,f1:0},identity:{reviewed_matched_pairs:0,unknown_or_unmatched_pairs:4,correct_pairs:0,pair_accuracy:null}};
const metadata=advancedMetadataSchema.parse({recording_id:'team_goal',selected_experiment:'m20_control',default_experiment:'m20_control',reference_role:'development_only',final_test_status:'pending',experiments:[{id:'m20_control',metrics:metric,aggregate:metric},{id:'offline_dag',metrics:metric,aggregate:metric}],camera:{},physics:{accepted:false,reason:'full camera required'},selection:{application_default_changed:false,final_accuracy_acceptance:'pending'}});
describe('M21 development comparisons',()=>{
 it('labels development metrics and leaves undefined precision unavailable',()=>{
  render(<AdvancedPerception metadata={metadata} video={null} width={640} height={360} fps={25} onExperiment={()=>{}} onSeek={()=>{}}/>);
  expect(screen.getByText(/new independent test labels are pending/i)).toBeInTheDocument();
  expect(screen.getByRole('link',{name:'Open new source-only review'})).toHaveAttribute('href','/?review=m21&recording=m21_holdout_1');
  expect(screen.queryByRole('img',{name:'M20 comparison video'})).not.toBeInTheDocument();
 });
 it('selects an explicit experiment without changing defaults',()=>{
  const change=vi.fn();render(<AdvancedPerception metadata={metadata} video={null} width={640} height={360} fps={25} onExperiment={change} onSeek={()=>{}}/>);
  fireEvent.change(screen.getByRole('combobox',{name:'M21 experiment'}),{target:{value:'offline_dag'}});
  expect(change).toHaveBeenCalledWith('offline_dag');expect(metadata.selection.application_default_changed).toBe(false);
 });
 it('never draws a stale-frame overlay',()=>{
  expect(overlayMatchesFrame(20,20)).toBe(true);expect(overlayMatchesFrame(20,19)).toBe(false);expect(overlayMatchesFrame(20,undefined)).toBe(false);
 });
 it('has readable experiment labels and a loading state',()=>{
  expect(experimentLabel('imm_multiframe')).toMatch(/IMM/);
  render(<AdvancedPerception metadata={null} video={null} width={640} height={360} fps={25} onExperiment={()=>{}} onSeek={()=>{}}/>);
  expect(screen.getByRole('status')).toHaveTextContent('Loading M21');
 });
});
