import {afterEach,beforeEach,describe,it,expect,vi} from 'vitest';
import {render,screen,waitFor,cleanup,fireEvent} from '@testing-library/react';import userEvent from '@testing-library/user-event';
import {IDENTITY_KEY,defaults} from './identity';
import App from './App';import {api} from './api';import {view,result} from './fixtures';
vi.mock('./api',()=>({api:{view:vi.fn(),ask:vi.fn(),player:vi.fn(),explain:vi.fn()},mediaUrl:(x:string)=>'/api'+x}));
beforeEach(()=>{localStorage.clear();localStorage.setItem(IDENTITY_KEY,JSON.stringify(defaults));vi.mocked(api.view).mockResolvedValue(structuredClone(view));vi.mocked(api.ask).mockResolvedValue(structuredClone(result));vi.mocked(api.player).mockResolvedValue({status:'ok',facts:result.answer.facts,limitations:[]});vi.mocked(api.explain).mockResolvedValue(result)});
afterEach(()=>{cleanup();vi.clearAllMocks()});
async function ready(){render(<App/>);await screen.findByText('API connected')}
describe('Match Intelligence',()=>{
 it('uses release pre-roll while retaining the canonical event marker and evidence time',async()=>{
  const data=structuredClone(view);data.events[0].metrics={start_frame:150,end_frame:190};
  vi.mocked(api.view).mockResolvedValue(data);await ready();
  const marker=screen.getByRole('button',{name:'Completed transfer at 00:08.00'});
  const position=marker.style.left;
  await userEvent.click(marker);
  expect(screen.getByTestId('current-time')).toHaveTextContent('00:04.00');
  expect(screen.getByRole('img',{name:'Tactical pitch at 4.00 seconds'})).toBeInTheDocument();
  expect(marker.style.left).toBe(position);expect(marker).toHaveClass('selected');
  expect(screen.getAllByRole('button',{name:'00:08.00'}).length).toBeGreaterThan(0);
  expect(data.events[0].timestamp).toBe(8);
 });
 it('Take me there seeks from verified conversation evidence without changing its timestamp',async()=>{
  const answer={...structuredClone(result),conversation:{recording_id:'118575_clip',last_timestamp:8,last_event_id:'evt_test',last_player_id:101}};
  vi.mocked(api.ask).mockResolvedValue(answer);await ready();
  await userEvent.type(screen.getByLabelText('Ask about this match'),'Take me there');
  await userEvent.click(screen.getByRole('button',{name:'Ask TactiqAI'}));
  await waitFor(()=>expect(screen.getByTestId('current-time')).toHaveTextContent('00:06.00'));
  expect(answer.conversation.last_timestamp).toBe(8);expect(answer.answer.statements[0].timestamp).toBe(8);
 });

 it('seeks canonical insights and preserves facts while switching presentation mode',async()=>{
  const data=structuredClone(view);const answer=structuredClone(result.answer);
  data.intelligence={recording_id:data.recording_id,limitations:[],insights:[{insight_id:'speed',title:'Fastest tracked player',classification:'INFERRED',timestamp:8,actor_id:101,event_id:null,presentations:{beginner:{...answer,mode:'beginner'},fan:{...answer,mode:'fan'},analyst:{...answer,mode:'analyst'}}}]};
  vi.mocked(api.view).mockResolvedValue(data);await ready();
  await userEvent.click(screen.getByText('Fastest tracked player · inferred'));
  await userEvent.click(screen.getByRole('button',{name:'Review 00:08.00'}));
  expect(screen.getByTestId('current-time')).toHaveTextContent('00:06.00');
  expect(screen.getByRole('img',{name:'Tactical pitch at 6.00 seconds'})).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button',{name:'Analyst'}));
  expect(screen.getByText('Canonical provenance')).toBeInTheDocument();
  expect(api.ask).not.toHaveBeenCalled();
 });

 it('renders the recording, unknown formation and missing media honestly',async()=>{await ready();expect(screen.getByRole('button',{name:'Match setup'})).toBeInTheDocument();expect(screen.getByText('Match video unavailable')).toBeInTheDocument();expect(screen.getAllByText(/Unknown/i).length).toBeGreaterThan(0);expect(screen.getByText('66.7%')).toBeInTheDocument()});
 it('shows completed-pass shares without frame-control percentages',async()=>{await ready();expect(screen.getByText('66.7%')).toBeInTheDocument();expect(screen.getByText('33.3%')).toBeInTheDocument();expect(screen.queryByText(/total confirmed control/)).not.toBeInTheDocument();expect(screen.getByText('Share of confirmed completed passes')).toBeInTheDocument()});
 it('uses continuity ownership and slow formations while pressure levels change independently',async()=>{
  const data=structuredClone(view);data.ownership=[{frame:0,timestamp:0,player_id:101,team:'left',attribution:'controlled'},{frame:125,timestamp:5,player_id:101,team:'left',attribution:'retained'},{frame:200,timestamp:8,player_id:102,team:'right',attribution:'controlled'}];
  data.formation_windows=[{team:'left',start_seconds:0,end_seconds:240,estimate:'4-3-3',reason:null}];data.pressure_levels=['high','low'];data.pressure_context=[{carrier_actor_id:101,attacking_team:'left',defenders_within_local:2},{carrier_actor_id:101,attacking_team:'left',defenders_within_local:0}];
  vi.mocked(api.view).mockResolvedValue(data);await ready();expect(screen.getByText('high',{exact:true})).toBeInTheDocument();
  fireEvent.change(screen.getByRole('slider',{name:'Match timestamp'}),{target:{value:'5'}});expect(screen.getByText(/Current possession · Home side · Player 1 · retained/)).toBeInTheDocument();expect(screen.getByText('4-3-3')).toBeInTheDocument();expect(screen.getByText('low',{exact:true})).toBeInTheDocument();
  fireEvent.change(screen.getByRole('slider',{name:'Match timestamp'}),{target:{value:'8'}});expect(screen.getByText('Current possession · Away side · Player 2')).toBeInTheDocument();expect(screen.getByText('4-3-3')).toBeInTheDocument();
 });
 it('accepts typed pass-count paraphrases and displays the grounded reply',async()=>{
  vi.mocked(api.ask).mockResolvedValue({...result,answer:{...result.answer,question_type:'possession_summary',statements:[{text:'Home side completed 16 observable transfers. Away side completed 13. Both teams completed 29 altogether.',fact_ids:[],timestamp:null,event_id:null}],facts:[]}});
  await ready();
  for(const question of ['How many passes did both teams make?','How many completed passes did each side have?','Who passed more?','Which team completed the most passes?','How many passes were there altogether?']){
   await userEvent.clear(screen.getByLabelText('Ask about this match'));await userEvent.type(screen.getByLabelText('Ask about this match'),question);await userEvent.click(screen.getByRole('button',{name:'Ask TactiqAI'}));
   await waitFor(()=>expect(vi.mocked(api.ask).mock.calls.at(-1)?.[0]).toBe(question));await screen.findByText(/Both teams completed 29 altogether/);expect(screen.queryByText('I don’t have enough evidence from this match to answer that reliably.')).not.toBeInTheDocument();
  }
 });
 it('asks, exposes evidence, jumps and highlights the supported actor',async()=>{await ready();await userEvent.click(screen.getByRole('button',{name:'Who was the fastest player?'}));await screen.findByText('Player 1; peak estimated speed: 20 km/h');await userEvent.click(screen.getByRole('button',{name:/Jump to moment/}));expect(screen.getByTestId('current-time')).toHaveTextContent('00:06.00');await screen.findByRole('heading',{name:'Player 1'});expect(api.ask).toHaveBeenCalledWith('Who was the fastest player?','fan','local',0,{});expect(screen.getByText(/Supporting evidence/)).toBeInTheDocument()});
 it('uses the same timestamp path for timeline events',async()=>{await ready();await userEvent.click(screen.getByRole('button',{name:'Completed transfer at 00:08.00'}));expect(screen.getByTestId('current-time')).toHaveTextContent('00:06.00');expect(screen.getByText('SELECTED EVENT')).toBeInTheDocument();await userEvent.click(screen.getByRole('button',{name:/Explain this event/}));expect(api.explain).toHaveBeenCalledWith('evt_test','fan');await screen.findByText('Event explanation')});
 it('requests backend presentation when mode changes',async()=>{await ready();await userEvent.click(screen.getByRole('button',{name:'Who was the fastest player?'}));await screen.findByText('Player 1; peak estimated speed: 20 km/h');await userEvent.click(screen.getByRole('button',{name:'Analyst'}));await waitFor(()=>expect(vi.mocked(api.ask).mock.calls.at(-1)?.[1]).toBe('analyst'))});
 it('scrubs the canonical timestamp directly',async()=>{await ready();fireEvent.change(screen.getByRole('slider',{name:'Match timestamp'}),{target:{value:'5'}});expect(screen.getByTestId('current-time')).toHaveTextContent('00:05.00');expect(screen.getByRole('img',{name:'Tactical pitch at 5.00 seconds'})).toBeInTheDocument()});
 it('handles API startup failure and recovery',async()=>{vi.mocked(api.view).mockRejectedValueOnce(new Error('API offline'));render(<App/>);await screen.findByText('Match data is unavailable');await userEvent.click(screen.getByRole('button',{name:/Retry connection/}));await screen.findByText('API connected')});
 it('displays fallback explicitly',async()=>{vi.mocked(api.ask).mockResolvedValue({...result,fallback:true});await ready();await userEvent.click(screen.getByRole('button',{name:'Who was the fastest player?'}));await screen.findByText('Local fallback');expect(screen.getByText(/local facts retained/)).toBeInTheDocument()});
 it('displays truthful unsupported answers',async()=>{vi.mocked(api.ask).mockResolvedValue({...result,answer:{...result.answer,status:'unsupported',statements:[{text:'Cannot determine this from tracking.',timestamp:null,event_id:null,fact_ids:[]}],facts:[]}});await ready();await userEvent.type(screen.getByLabelText('Ask about this match'),'What did the coach say?');await userEvent.click(screen.getByRole('button',{name:'Ask TactiqAI'}));await screen.findByText('I don’t have enough evidence from this match to answer that reliably.');expect(screen.queryByRole('button',{name:/Jump to moment/})).not.toBeInTheDocument()});
 it('keeps tactical inspection available when Ask fails',async()=>{vi.mocked(api.ask).mockRejectedValue(new Error('The request timed out.'));await ready();await userEvent.click(screen.getByRole('button',{name:'Who was the fastest player?'}));await screen.findByRole('alert');expect(screen.getByRole('slider')).toBeEnabled()});
 it('filters empty categories without inventing events',async()=>{await ready();await userEvent.selectOptions(screen.getByLabelText('Filter timeline'),'shots');expect(screen.getByText('No canonical events in this category.')).toBeInTheDocument()});
});
