import {afterEach,it,expect,vi} from 'vitest';
import {cleanup,render,screen,fireEvent,waitFor} from '@testing-library/react';
import M24Benchmark from './M24Benchmark';
afterEach(()=>{cleanup();vi.restoreAllMocks();history.replaceState({},'', '/')});
const status={correct:false,missed:true,false:false,prediction:null};
const meta={schema_version:'cv-m24-browser-v1',recording_id:'giroud',frames:450,fps:25,width:640,height:360,recordings:['giroud'],video_url:'/cv/m24/matches/giroud/video.mp4',read_only:true,threshold:.7,tolerance_diagonal_fraction:.01,reference_revision:'a'.repeat(64),navigation:{giroud_completed:{reviewed:[60,90],missed:[60,90],correct:[],false:[],fa_wins:[],m23_wins:[],both_fail:[60,90]}},groups:{giroud_completed:{footandball:{primary:{tp:0,fp:0,fn:31,precision:null,recall:0,f1:0}}}}};
function mock(){return vi.spyOn(globalThis,'fetch').mockImplementation(async input=>{const path=String(input);return new Response(JSON.stringify(path.endsWith('/metadata')?meta:{frame:Number(new URL(path,'http://localhost').searchParams.get('frame')),timestamp:0,footandball:{candidates:[{center:[20,30],confidence:.1}],selected:null},m23:{selected:null},references:{giroud_completed:{state:'visible',bbox:[10,10,5,5],tolerance_pixels:7.3,status:{footandball:status,m23:status}}}}),{status:200})})}
it('keeps low-confidence candidates separate from selected observations and never writes labels',async()=>{
 const fetch=mock();render(<M24Benchmark/>);expect(await screen.findByText('Missed ball')).toBeTruthy();expect(screen.getAllByText('No selected detection')).toHaveLength(2);expect(screen.queryByRole('button',{name:/save|accept/i})).toBeNull();expect(fetch.mock.calls.every(([,init])=>!init?.method||init.method==='GET')).toBe(true);
});
it('navigates reviewed cases and enables a mapped heatmap without modifying references',async()=>{
 mock();render(<M24Benchmark/>);await screen.findByText('Missed ball');fireEvent.click(screen.getByRole('button',{name:'Next case'}));await waitFor(()=>expect(screen.getByLabelText('Benchmark frame')).toHaveValue(60));fireEvent.click(screen.getByLabelText('Heatmap'));expect(screen.getByAltText('Ball confidence heatmap')).toHaveAttribute('src',expect.stringContaining('frame=60'));
});
it('handles a missing optional benchmark with a useful error and workspace link',async()=>{
 vi.spyOn(globalThis,'fetch').mockResolvedValue(new Response('{}',{status:503}));render(<M24Benchmark/>);expect(await screen.findByRole('alert')).toHaveTextContent('503');expect(screen.getByRole('link',{name:/Match workspace/})).toHaveAttribute('href','/?recording=giroud');
});
