import {render,screen,fireEvent,cleanup,waitFor} from '@testing-library/react';
import {afterEach,it,expect,vi} from 'vitest';
import M22HumanReview from './M22HumanReview';
const task=(id:string,frame:number)=>({task_id:id,frame,timestamp:frame/25,sequence_id:'seq',sequence_frames:[10,11],proposals:[{proposal_id:'p'+id,bbox:[20,30,8,8],confidence:.6,source_pipeline:'m21'}],selection_reason:'detector disagreement'});
const initial={dataset_fingerprint:'dev',reference_role:'development_only',model_proposals_are_truth:false,source:{width:640,height:360,fps:25,video_sha256:'a'.repeat(64)},tasks:[task('one',10),task('two',11)],labels:[] as any[],sources:[{recording_id:'test_match',tasks:2,saved_tasks:0}]};
afterEach(()=>{cleanup();vi.unstubAllGlobals();history.replaceState({},'', '/')});
function setup({saved=false,fail=false}={}){
 const data=structuredClone(initial);if(saved){data.labels=[{task_id:'one',reviewer_id:'AB',state:'visible',bbox:[20,30,8,8],accepted_proposal:'pone',rejected_proposals:[],notes:'Saved earlier'}];data.sources[0].saved_tasks=1}
 const fetch=vi.fn().mockImplementation(async(_url,options)=>{
  if(options?.method==='POST'){
   if(fail)return {ok:false,json:async()=>({detail:'Check the source box'})};
   const label=JSON.parse(options.body);data.labels=[...data.labels.filter(l=>l.task_id!==label.task_id),label];data.sources[0].saved_tasks=data.labels.length;
   return {ok:true,json:async()=>({saved:true,task_id:label.task_id})};
  }
  return {ok:true,json:async()=>structuredClone(data)};
 });vi.stubGlobal('fetch',fetch);render(<M22HumanReview/>);return fetch;
}
it('shows only approved short-clip source frames and never autosaves model proposals',async()=>{
 const fetch=setup();await screen.findByRole('region',{name:'M22 ball annotation'});
 expect(screen.getByRole('img',{name:'Current source frame: verify the football'}).querySelector('image')).toHaveAttribute('href','/api/cv/m22/matches/test_match/review/frame.jpg?frame=10');
 expect(screen.getByRole('button',{name:'Save verified development label'})).toBeDisabled();
 fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));fireEvent.change(screen.getByLabelText('Notes'),{target:{value:'still editing'}});
 expect(fetch.mock.calls.filter(c=>c[1]?.method==='POST')).toHaveLength(0);
 expect(screen.queryByText(/holdout/)).toBeNull();
});
it('requires explicit verification, saves assisted provenance and refreshes progress',async()=>{
 const fetch=setup();await screen.findByRole('region',{name:'M22 ball annotation'});
 fireEvent.change(screen.getByLabelText('Reviewer ID'),{target:{value:'AB'}});fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));
 expect(screen.getByRole('button',{name:'Save verified development label'})).toBeDisabled();fireEvent.click(screen.getByLabelText(/I checked the actual source/));
 fireEvent.click(screen.getByRole('button',{name:'Save verified development label'}));await screen.findByRole('status');
 expect(screen.getByTestId('m22-review-progress')).toHaveTextContent('1/2');const sent=JSON.parse(fetch.mock.calls.find(c=>c[1]?.method==='POST')![1].body);
 expect(sent.model_assisted).toBe(true);expect(sent.origin).toBe('human');expect(sent.accepted_proposal).toBe('pone');expect(sent.state).toBe('visible');
});
it('restores saved annotations when revisiting frames and starts at the first unsaved task',async()=>{
 setup({saved:true});await screen.findByText(/Frame 11 ·/);fireEvent.click(screen.getByRole('button',{name:'Previous selected frame'}));
 await screen.findByText(/Frame 10 ·/);expect(screen.getByLabelText('Ball visibility')).toHaveValue('visible');expect(screen.getByLabelText('Notes')).toHaveValue('Saved earlier');
});
it('keeps a rejected proposal separate from whole-frame absence',async()=>{
 const fetch=setup();await screen.findByRole('region',{name:'M22 ball annotation'});fireEvent.change(screen.getByLabelText('Reviewer ID'),{target:{value:'AB'}});
 fireEvent.click(screen.getByRole('button',{name:'Not the ball 1'}));expect(screen.getByLabelText('Ball visibility')).toHaveValue('uncertain');
 fireEvent.click(screen.getByLabelText(/I checked the actual source/));fireEvent.click(screen.getByRole('button',{name:'Save verified development label'}));await screen.findByRole('status');
 const sent=JSON.parse(fetch.mock.calls.find(c=>c[1]?.method==='POST')![1].body);expect(sent.bbox).toBeNull();expect(sent.rejected_proposals).toEqual(['pone']);expect(sent.state).toBe('uncertain');
});
it('preserves edits after save rejection and prevents accidental unsaved navigation',async()=>{
 setup({fail:true});await screen.findByRole('region',{name:'M22 ball annotation'});fireEvent.change(screen.getByLabelText('Reviewer ID'),{target:{value:'AB'}});fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));
 fireEvent.click(screen.getByRole('button',{name:'Next selected frame'}));await screen.findByRole('alert');expect(screen.getByText(/Frame 10 ·/)).toBeInTheDocument();
 fireEvent.click(screen.getByLabelText(/I checked the actual source/));fireEvent.click(screen.getByRole('button',{name:'Save verified development label'}));await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('Check the source box'));expect(screen.getByLabelText('Ball visibility')).toHaveValue('visible');
});
