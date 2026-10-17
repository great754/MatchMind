import {describe,it,expect} from 'vitest';
import {replayIndex,tacticalAt,eventLabel,filterEvents,numberLabel,timeLabel} from './time';
import {view,event} from './fixtures';
describe('timestamp synchronization',()=>{
 it('selects the preceding supported sample and clamps boundaries',()=>{expect(replayIndex(view.replay!,0)).toBe(0);expect(replayIndex(view.replay!,4.99)).toBe(0);expect(replayIndex(view.replay!,5)).toBe(1);expect(replayIndex(view.replay!,1000)).toBe(1)});
 it('does not show a future tactical state',()=>{expect(tacticalAt(view.tactical_states,-1)).toBeNull();expect(tacticalAt(view.tactical_states,4.9)?.timestamp).toBe(0);expect(tacticalAt(view.tactical_states,5)?.timestamp).toBe(5);expect(tacticalAt([],5)).toBeNull()});
 it('retains shot-candidate semantics',()=>expect(eventLabel({...event,type:'SHOT_CANDIDATE'})).toBe('Shot candidate'));
 it('filters without inferring new events',()=>{expect(filterEvents([event],'transfers')).toEqual([event]);expect(filterEvents([event],'shots')).toEqual([])});
 it('renders missing measurements explicitly',()=>{expect(numberLabel(null,' m')).toBe('Unavailable');expect(timeLabel(171.28)).toBe('02:51')});
});
