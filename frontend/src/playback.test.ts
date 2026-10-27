import {describe,it,expect} from 'vitest';
import {getPlaybackAnchorTime,getPlaybackSeekTime} from '../../src/serving/static/playback.js';
const event={timestamp:11.8,source:'transfer',metrics:{start_frame:260,end_frame:290}};
describe('shared presentation playback policy',()=>{
 it('uses release, not receipt/confirmation; preserves canonical input',()=>{const original=structuredClone(event);expect(getPlaybackAnchorTime(11.8,event,25)).toBe(10.4);expect(getPlaybackSeekTime(11.8,event,25,30)).toBe(8.4);expect(event).toEqual(original)});
 it('uses candidate start for shots',()=>{expect(getPlaybackSeekTime(11.8,{...event,source:'trajectory'},25,30)).toBe(8.4)});
 it('uses recorded pressure/tactical evidence start',()=>{expect(getPlaybackSeekTime(11.8,{...event,source:'tactical_state_engine',tactical:{evidence_start_frame:200}},25,30)).toBe(6)});
 it('falls back without guessing formation/control onset',()=>{for(const source of ['possession','bas','motion','tactical_state_engine'])expect(getPlaybackSeekTime(10,{timestamp:10,source},25,30)).toBe(8)});
 it('clamps early moments, duration and bad input',()=>{expect(getPlaybackSeekTime(.5,null,25,30)).toBe(0);expect(getPlaybackSeekTime(10,null,25,30)).toBe(8);expect(getPlaybackSeekTime(50,null,25,12)).toBe(12);expect(getPlaybackSeekTime(NaN,null,25,12)).toBe(0);expect(getPlaybackSeekTime(10,event,0,30)).toBe(8)});
 it('rejects future/negative starts and supports one bounded configurable pre-roll',()=>{expect(getPlaybackSeekTime(11.8,{...event,metrics:{start_frame:500}},25,30)).toBe(9.8);expect(getPlaybackSeekTime(10,null,25,30,1)).toBe(9);expect(getPlaybackSeekTime(10,null,25,30,-1)).toBe(10)});
});
