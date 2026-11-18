import {it,expect,vi,afterEach} from 'vitest';
import {render,screen,cleanup,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RecordingsApp from './RecordingsApp';
import {recordingList} from './api';
import {loadIdentity,saveIdentity,defaults} from './identity';
vi.mock('./api',()=>({recordingList:vi.fn(),mediaUrl:(path:string)=>'/api'+path}));
vi.mock('./App',()=>({default:({recordingId,recordingPicker}:{recordingId:string,recordingPicker:React.ReactNode})=><div>{recordingPicker}<p>Showing {recordingId}</p></div>}));
vi.mock('./CVReview',()=>({default:({recordingId,recordingPicker}:{recordingId:string,recordingPicker:React.ReactNode})=><div>{recordingPicker}<p>CV review {recordingId}</p></div>}));
afterEach(()=>{cleanup();vi.clearAllMocks();localStorage.clear();window.history.replaceState({},'','/')});
it('switches independent processed recordings and disables pending recordings',async()=>{vi.mocked(recordingList).mockResolvedValue({recordings:[{recording_id:'118575_clip',match_id:'118575',status:'available'},{recording_id:'128058_clip',match_id:'128058',status:'available'},{recording_id:'pending_clip',match_id:'pending',status:'not_processed'}]});render(<RecordingsApp/>);await screen.findByRole('option',{name:'128058_clip'});await userEvent.selectOptions(screen.getByLabelText('Recording'),'128058_clip');expect(screen.getByText('Showing 128058_clip')).toBeInTheDocument();expect(screen.getByRole('option',{name:/pending_clip/})).toBeDisabled()});
it('isolates local identities by recording',()=>{saveIdentity({...defaults,teams:{...defaults.teams,left:{name:'Old club',color:'#112233'}}},'118575_clip');expect(loadIdentity('128058_clip').teams.left.name).toBe('Home side');expect(loadIdentity('118575_clip').teams.left.name).toBe('Old club')});
it('allows video-only selection while withholding unavailable analytics',async()=>{vi.mocked(recordingList).mockResolvedValue({recordings:[{recording_id:'118575_clip',match_id:'118575',status:'available'},{recording_id:'117092_clip',match_id:'117092',status:'unavailable',media_available:true,media_url:'/matches/117092_clip/video.mp4'}]});render(<RecordingsApp/>);const option=await screen.findByRole('option',{name:/117092_clip.*video ready/});expect(option).toBeEnabled();await userEvent.selectOptions(screen.getByLabelText('Recording'),'117092_clip');expect(screen.getByText('Match analysis is not available yet')).toBeInTheDocument();expect(document.querySelector('video')).toHaveAttribute('src','/api/matches/117092_clip/video.mp4');expect(screen.queryByText('Showing 117092_clip')).not.toBeInTheDocument();await userEvent.selectOptions(screen.getByLabelText('Recording'),'118575_clip');expect(screen.getByText('Showing 118575_clip')).toBeInTheDocument()});
it('disables missing and invalid Azure snapshots while selecting a usable recording',async()=>{vi.mocked(recordingList).mockResolvedValue({recordings:[{recording_id:'118575_clip',match_id:'118575',status:'unavailable'},{recording_id:'128058_clip',match_id:'128058',status:'available'},{recording_id:'bad',match_id:'bad',status:'error'}]});render(<RecordingsApp/>);await screen.findByText('Showing 128058_clip');expect(screen.getByRole('option',{name:/118575_clip.*artifacts unavailable/})).toBeDisabled();expect(screen.getByRole('option',{name:/bad.*artifact error/})).toBeDisabled()});
it('keeps the default recording available if discovery fails',async()=>{vi.mocked(recordingList).mockRejectedValue(new Error('offline'));render(<RecordingsApp/>);await waitFor(()=>expect(screen.getByLabelText('Recording')).toHaveAttribute('title','Recording list unavailable; showing the default recording.'));expect(screen.getByText('Showing 118575_clip')).toBeInTheDocument()});

it('uses capabilities to offer both analytics and CV review without recording-ID rules',async()=>{
 vi.mocked(recordingList).mockResolvedValue({recordings:[
  {recording_id:'118575_clip',match_id:'118575',status:'available',capabilities:{analytics_available:true,ask_available:true,cv_world_model_available:true}},
  {recording_id:'arbitrary_cv_source',match_id:'other',status:'available',source:'cv_experimental',capabilities:{analytics_available:false,ask_available:false,cv_world_model_available:true}}
 ]});
 render(<RecordingsApp/>);await screen.findByLabelText('Review mode');expect(screen.getByText('Showing 118575_clip')).toBeInTheDocument();
 await userEvent.selectOptions(screen.getByLabelText('Review mode'),'cv');expect(screen.getByText('CV review 118575_clip')).toBeInTheDocument();
 await userEvent.selectOptions(screen.getByLabelText('Review mode'),'analytics');expect(screen.getByText('Showing 118575_clip')).toBeInTheDocument();
 await userEvent.selectOptions(screen.getByLabelText('Recording'),'arbitrary_cv_source');expect(screen.getByText('CV review arbitrary_cv_source')).toBeInTheDocument();expect(screen.queryByLabelText('Review mode')).not.toBeInTheDocument();
});

it('waits for recording capabilities before mounting analytics for a CV deep link',async()=>{
 window.history.replaceState({},'', '/?recording=test_match');
 let resolve!:(value:Awaited<ReturnType<typeof recordingList>>)=>void;
 vi.mocked(recordingList).mockImplementation(()=>new Promise(done=>{resolve=done}));
 render(<RecordingsApp/>);expect(screen.getByRole('status')).toHaveTextContent('Loading recordings');expect(screen.queryByText('Showing test_match')).not.toBeInTheDocument();
 resolve({recordings:[{recording_id:'test_match',match_id:'test_match',status:'available',source:'cv_experimental',capabilities:{cv_world_model_available:true,analytics_available:false,ask_available:false}}]});
 await screen.findByText('CV review test_match');expect(screen.queryByText('Showing test_match')).not.toBeInTheDocument();
});
it('keeps an undiscovered deep link out of analytics and allows discovery retry',async()=>{
 window.history.replaceState({},'', '/?recording=test_match');vi.mocked(recordingList).mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce({recordings:[{recording_id:'test_match',match_id:'test_match',status:'available',source:'cv_experimental',capabilities:{cv_world_model_available:true,analytics_available:false,ask_available:false}}]});
 render(<RecordingsApp/>);await screen.findByRole('alert');expect(screen.queryByText('Showing test_match')).not.toBeInTheDocument();await userEvent.click(screen.getByRole('button',{name:'Retry loading recordings'}));await screen.findByText('CV review test_match');
});
