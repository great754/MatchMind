import {render,screen,fireEvent,cleanup,waitFor} from '@testing-library/react';
import {afterEach,beforeEach,it,expect,vi} from 'vitest';
import M22HumanReview from './M22HumanReview';
const proposal=(id:string,bbox=[20,30,8,8])=>({proposal_id:id,bbox,confidence:.6,source_pipeline:'m21'});
const task=(id:string,frame:number)=>({task_id:id,frame,timestamp:frame/25,sequence_id:'seq',sequence_frames:[10,11],proposals:[proposal('p'+id)],selection_reason:'detector disagreement'});
const initial={dataset_fingerprint:'dev',reference_role:'development_only',model_proposals_are_truth:false,supports_reviewed_proposals:true,source:{width:640,height:360,fps:25,video_sha256:'a'.repeat(64)},tasks:[task('one',10),task('two',11)],labels:[] as any[],sources:[{recording_id:'test_match',tasks:2,saved_tasks:0}]};
beforeEach(()=>{localStorage.clear();localStorage.setItem('tactiq.m22.reviewer','AB')});
afterEach(()=>{cleanup();vi.unstubAllGlobals();history.replaceState({},'', '/')});
function setup({saved=false,fail=false,extra=false,duplicate=false,hold=false}={}){
 const data=structuredClone(initial);if(extra||duplicate)data.tasks[0].proposals.push(proposal('other',duplicate?[21,30,8,8]:[70,40,8,8]));
 if(saved){data.labels=[{task_id:'one',reviewer_id:'AB',state:'visible',bbox:[20,30,8,8],accepted_proposal:'pone',rejected_proposals:[],notes:'Saved earlier'}];data.sources[0].saved_tasks=1}
 let release:((value:any)=>void)|null=null;
 const fetch=vi.fn().mockImplementation(async(_url,options)=>{
  if(options?.method==='POST'){
   if(fail)return {ok:false,json:async()=>({detail:'Check the source box'})};
   const label=JSON.parse(options.body);data.labels=[...data.labels.filter(l=>l.task_id!==label.task_id),label];data.sources[0].saved_tasks=data.labels.length;
   const response={ok:true,json:async()=>({saved:true,task_id:label.task_id})};
   return hold?new Promise(resolve=>{release=resolve}):response;
  }
  return {ok:true,json:async()=>structuredClone(data)};
 });vi.stubGlobal('fetch',fetch);render(<M22HumanReview/>);
 return {fetch,posts:()=>fetch.mock.calls.filter(c=>c[1]?.method==='POST'),release:()=>release?.({ok:true,json:async()=>({saved:true,task_id:'one'})})};
}
async function ready(){await screen.findByRole('region',{name:'M22 ball annotation'});fireEvent.load(screen.getByRole('img',{name:'Current source frame: verify the football'}).querySelector('image')!)}
it('does not save suggestions on open and keeps advanced controls collapsed',async()=>{
 const state=setup();await ready();expect(state.posts()).toHaveLength(0);expect(screen.queryByRole('button',{name:/Save verified/})).toBeNull();
 expect(screen.getByLabelText('Notes').closest('details')).not.toHaveAttribute('open');expect(screen.queryByText(/holdout/)).toBeNull();expect(screen.getByRole('button',{name:'Accept proposal 1'})).toBeEnabled();
});
it('automatically saves reviewed provenance and advances without a Save click',async()=>{
 const state=setup();await ready();fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));
 await screen.findByText(/Frame 11 ·/);expect(state.posts()).toHaveLength(1);
 const sent=JSON.parse(state.posts()[0][1].body);expect(sent).toMatchObject({task_id:'one',reviewer_id:'AB',model_assisted:true,origin:'human',reviewed_from_source:true,accepted_proposal:'pone',reviewed_proposals:['pone'],state:'visible'});
 expect(screen.getByTestId('m22-review-progress')).toHaveTextContent('1/2');
});
it('waits until all proposals have a decision, then saves once',async()=>{
 const state=setup({extra:true});await ready();fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));
 await new Promise(resolve=>setTimeout(resolve,700));expect(state.posts()).toHaveLength(0);
 fireEvent.click(screen.getByRole('button',{name:'Not the ball 2'}));await screen.findByText(/Frame 11 ·/);
 expect(state.posts()).toHaveLength(1);const sent=JSON.parse(state.posts()[0][1].body);expect(sent.rejected_proposals).toEqual(['other']);expect(sent.reviewed_proposals).toEqual(['pone','other']);
});
it('waits for server acknowledgement and disables duplicate writes/navigation in flight',async()=>{
 const state=setup({hold:true});await ready();fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));
 await waitFor(()=>expect(state.posts()).toHaveLength(1));expect(screen.getByText(/Frame 10 ·/)).toBeInTheDocument();expect(screen.getByRole('button',{name:'Next frame'})).toBeDisabled();
 fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));expect(state.posts()).toHaveLength(1);state.release();await screen.findByText(/Frame 11 ·/);
});
it('restores saved decisions on back, and automatically saves a correction',async()=>{
 const state=setup();await ready();fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));await screen.findByText(/Frame 11 ·/);
 fireEvent.click(screen.getByRole('button',{name:'Previous frame'}));await ready();expect(screen.getByRole('button',{name:'Accept proposal 1'})).toHaveAttribute('aria-pressed','true');
 expect(state.posts()).toHaveLength(1);fireEvent.click(screen.getByRole('button',{name:"Can't tell"}));await screen.findByText(/Frame 11 ·/);
 expect(state.posts()).toHaveLength(2);expect(JSON.parse(state.posts()[1][1].body).state).toBe('uncertain');
});
it('does not turn rejected proposals into an absent ball without an explicit visibility choice',async()=>{
 const state=setup();await ready();fireEvent.click(screen.getByRole('button',{name:'Not the ball 1'}));
 await new Promise(resolve=>setTimeout(resolve,700));expect(state.posts()).toHaveLength(0);
 fireEvent.click(screen.getByRole('button',{name:"Can't tell"}));await screen.findByText(/Frame 11 ·/);
 const sent=JSON.parse(state.posts()[0][1].body);expect(sent.state).toBe('uncertain');expect(sent.bbox).toBeNull();expect(sent.rejected_proposals).toEqual(['pone']);
});
it('supports overlapping positive proposals without making a false hard negative',async()=>{
 const state=setup({duplicate:true});await ready();fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));expect(screen.getByRole('button',{name:'Not the ball 2'})).toBeDisabled();
 fireEvent.click(screen.getByRole('button',{name:'Accept proposal 2'}));await screen.findByText(/Frame 11 ·/);
 const sent=JSON.parse(state.posts()[0][1].body);expect(sent.reviewed_proposals).toEqual(['pone','other']);expect(sent.rejected_proposals).toEqual([]);
});
it('keeps failed saves on the current frame and retries only on request',async()=>{
 const state=setup({fail:true});await ready();fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));await screen.findByText('Check the source box');
 expect(screen.getByText(/Frame 10 ·/)).toBeInTheDocument();expect(screen.getByRole('button',{name:'Accept proposal 1'})).toHaveAttribute('aria-pressed','true');
 await new Promise(resolve=>setTimeout(resolve,700));expect(state.posts()).toHaveLength(1);fireEvent.click(screen.getByRole('button',{name:'Retry automatic save'}));await waitFor(()=>expect(state.posts()).toHaveLength(2));
});
it('keeps incomplete choices across navigation without posting partial labels',async()=>{
 const state=setup({extra:true});await ready();fireEvent.click(screen.getByRole('button',{name:'Accept proposal 1'}));fireEvent.click(screen.getByRole('button',{name:'Next frame'}));
 fireEvent.click(screen.getByRole('button',{name:'Previous frame'}));await ready();expect(screen.getByRole('button',{name:'Accept proposal 1'})).toHaveAttribute('aria-pressed','true');expect(state.posts()).toHaveLength(0);
});
it('resumes legacy saved labels without rewriting them and stays on the last question after saving',async()=>{
 const state=setup({saved:true});await ready();expect(screen.getByText(/Frame 11 ·/)).toBeInTheDocument();expect(state.posts()).toHaveLength(0);
 fireEvent.click(screen.getByRole('button',{name:'Ball not visible'}));await screen.findByText(/This clip is complete/);expect(screen.getByText(/Frame 11 ·/)).toBeInTheDocument();expect(screen.getByRole('button',{name:'Next frame'})).toBeDisabled();
});
it('does not save or allow proposal choices before the source image loads',async()=>{
 const state=setup();await screen.findByRole('region',{name:'M22 ball annotation'});expect(screen.getByRole('button',{name:'Accept proposal 1'})).toBeDisabled();expect(screen.getByRole('button',{name:'Ball not visible'})).toBeDisabled();expect(state.posts()).toHaveLength(0);
});
it('returns to an unfinished question instead of claiming completion after skipping it',async()=>{
 setup();await ready();fireEvent.click(screen.getByRole('button',{name:'Next frame'}));await ready();
 fireEvent.click(screen.getByRole('button',{name:"Can't tell"}));await screen.findByText(/Frame 10 ·/);
 expect(screen.queryByText(/This clip is complete/)).toBeNull();expect(screen.getByTestId('m22-review-progress')).toHaveTextContent('1/2');
});
