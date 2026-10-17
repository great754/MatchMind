import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import {render,screen,cleanup} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {IdentityContext,useIdentity,defaults,loadIdentity,saveIdentity,IDENTITY_KEY} from './identity';
import {Pitch} from './Pitch';
import {Evidence} from './Evidence';
import {view,result} from './fixtures';
import App from './App';
import {api} from './api';
vi.mock('./api',()=>({api:{view:vi.fn(),ask:vi.fn(),player:vi.fn(),explain:vi.fn()},mediaUrl:(v:string)=>v}));
beforeEach(()=>{localStorage.clear();vi.mocked(api.view).mockResolvedValue(structuredClone(view))});afterEach(()=>{cleanup();vi.clearAllMocks()});
const mapped={...defaults,teams:{left:{name:'Berry FC',color:'#11aa33'},right:{name:'Oak FC',color:'#9955ff'}},players:{'101':{name:'Marcus',jersey:'8',team:'left' as const}}};
function Names(){const n=useIdentity();return <p>{n.text('Actor 101 on Home side')} · {n.player(102)} · {n.canonicalQuestion('How far did Marcus run for Berry FC?')}</p>}
it('persists validated match labels, rejecting malformed stored data',()=>{expect(saveIdentity(mapped)).toBe(true);expect(loadIdentity()).toEqual(mapped);localStorage.setItem(IDENTITY_KEY,'{"bad":true}');expect(loadIdentity()).toEqual(defaults)});
it('resolves names, jerseys, teams and canonical questions without changing IDs',()=>{render(<IdentityContext.Provider value={{identity:mapped,ids:[101,102]}}><Names/></IdentityContext.Provider>);expect(screen.getByText('Marcus #8 on Berry FC · Player 2 · How far did player 101 run for Team 1?')).toBeInTheDocument()});
it('uses user-selected player team colors on the pitch',()=>{const {container}=render(<IdentityContext.Provider value={{identity:mapped,ids:[101,102]}}><Pitch data={view.replay} time={0} selected={101} onSelect={()=>{}} event={null} state={null}/></IdentityContext.Provider>);expect(screen.getByRole('button',{name:'Select Marcus #8'})).toBeInTheDocument();expect(container.querySelector('circle[fill="#11aa33"]')).toBeTruthy()});
it('keeps raw evidence out of every normal explanation mode',()=>{for(const mode of ['beginner','fan','analyst'] as const){const {container,unmount}=render(<IdentityContext.Provider value={{identity:mapped,ids:[101,102]}}><Evidence answer={{...result.answer,mode}} agent={result} mode={mode} onJump={()=>{}}/></IdentityContext.Provider>);expect(container.textContent).toContain('Marcus #8');for(const raw of ['Actor 101','player_id','f_speed','player_summary.csv#record=101'])expect(container.textContent).not.toContain(raw);unmount()}});
it('prompts first-time setup and saves team names and colors',async()=>{render(<App/>);await screen.findByRole('dialog');const names=screen.getAllByLabelText('Team name');await userEvent.clear(names[0]);await userEvent.type(names[0],'Berry FC');await userEvent.click(screen.getByRole('button',{name:'Save match setup'}));expect(screen.queryByRole('dialog')).not.toBeInTheDocument();expect(loadIdentity().teams.left.name).toBe('Berry FC');expect(screen.getAllByText('Berry FC').length).toBeGreaterThan(0)});
