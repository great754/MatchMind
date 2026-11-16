import {render,screen,fireEvent,cleanup,within} from '@testing-library/react';
import {afterEach,it,expect,vi} from 'vitest';
import CVPassPanel from './CVPassPanel';
const data={recording_id:'r',version:'cv-passes-v1',ball_version:'m14',statistics:{completed_pass_estimates:0,transfer_candidates:1,ball_observed_fraction:.25,goal_count:null,shot_count:null,pass_accuracy:null},events:[{event_id:'c',status:'candidate',sender_ref:'segment:a',receiver_ref:'segment:b',sender_label:'Player A',receiver_label:'Player B',start:6,end:7,ball_observed_fraction:.25,reasons:['BALL_OBSERVATION_GAP']}],players:[{player_ref:'segment:a',label:'Player A',entity_id:'entity-a'},{player_ref:'segment:b',label:'Player B',entity_id:'entity-b'}],limitations:['Unknown goals are not zero.']};
afterEach(()=>{cleanup();vi.unstubAllGlobals()});
it('keeps candidates out of pass counts and does not invent goal totals',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>data}));const open=vi.fn(),review=vi.fn();render(<CVPassPanel recordingId="r" focusedRef={null} onFocus={()=>{}} onOpen={open} onReview={review}/>);
 await screen.findByText('M14 ball evidence');expect(screen.getByTestId('cv-pass-count')).toHaveTextContent('0');expect(screen.getByTestId('cv-pass-candidates')).toHaveTextContent('1');expect(screen.getByText('Unavailable')).toBeVisible();expect(screen.getByText(/does not mean/)).toBeVisible();fireEvent.click(screen.getByText('Review 1 possible transfers'));fireEvent.click(screen.getByRole('button',{name:/Player A → Player B/}));expect(open).toHaveBeenCalledWith(3);fireEvent.click(screen.getByRole('button',{name:'Create a candidate review video'}));expect(review).toHaveBeenCalledOnce();
});
it('passes the selected segment and entity to the shared player focus',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>data}));const focus=vi.fn();render(<CVPassPanel recordingId="r" focusedRef={null} onFocus={focus} onOpen={()=>{}} onReview={()=>{}}/>);await screen.findByText('M14 ball evidence');fireEvent.change(screen.getByLabelText('CV player focus'),{target:{value:'segment:b'}});expect(focus).toHaveBeenCalledWith('segment:b','entity-b');
});
it('shows unavailable rather than zero when event validation fails',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false,status:503}));render(<CVPassPanel recordingId="r" focusedRef={null} onFocus={()=>{}} onOpen={()=>{}} onReview={()=>{}}/>);await screen.findByText('Pass analysis is not available for this recording yet.');expect(screen.queryByTestId('cv-pass-count')).toBeNull();
});
it('loads opt-in M19 separately and resets counts when switching versions',async()=>{
 const fetch=vi.fn().mockResolvedValue({ok:true,json:async()=>data});vi.stubGlobal('fetch',fetch);const props={recordingId:'r',focusedRef:null,onFocus:()=>{},onOpen:()=>{},onReview:()=>{}};const {rerender}=render(<CVPassPanel {...props}/>);await screen.findByText('M14 ball evidence');fetch.mockResolvedValue({ok:true,json:async()=>({...data,ball_version:'m19',events:[]})});rerender(<CVPassPanel {...props} cvPipeline="m19"/>);await screen.findByText('M19 ball evidence');expect(fetch.mock.lastCall?.[0]).toContain('ball_pipeline=m19');expect(screen.getByTestId('cv-pass-candidates')).toHaveTextContent('0');
});
