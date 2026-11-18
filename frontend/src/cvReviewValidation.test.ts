import {describe,it,expect} from 'vitest';
import {reviewProblems,reviewServerError,sourceReviewFrame} from './cvReviewValidation';
import type {ReviewLabel,ReviewTask} from './cvReviewValidation';
const source={fps:25,width:640,height:360,frames:100};
const task:ReviewTask={task_id:'ball',kind:'ball',frame:10};
const label=(extra:Partial<ReviewLabel>={}):ReviewLabel=>({task_id:'ball',reviewer_id:'AB',origin:'human',reviewed_from_source:true,model_assisted:false,state:'visible',bbox:[10,20,4,4],other_bbox:null,exhaustive:false,passes:[],notes:'',...extra});
describe('review save preflight',()=>{
 it('requires a visible ball box before any save',()=>{expect(reviewProblems(task,source,label({bbox:null}))).toContain('Draw a box around the visible ball.')});
 it('rejects zero-area, nonfinite and out-of-source rectangles',()=>{for(const bbox of [[10,20,0,4],[10,20,4,0],[-1,20,4,4],[639,20,4,4],[NaN,20,4,4]] as [number,number,number,number][]){expect(reviewProblems(task,source,label({bbox})).some(p=>p.startsWith('Redraw'))).toBe(true)}});
 it('requires both boxes for a positive identity comparison',()=>{const identity:ReviewTask={task_id:'i',kind:'identity',frame:10,other_frame:15};expect(reviewProblems(identity,source,label({state:'same',other_bbox:null}))).toContain('Draw a box around the compared player in the second image.');expect(reviewProblems(identity,source,label({state:'same',other_bbox:[30,20,20,60]}))).toEqual([])});
 it('keeps uncertainty valid without manufacturing boxes or a negative',()=>{expect(reviewProblems({...task,kind:'identity'},source,label({state:'uncertain',bbox:null}))).toEqual([]);expect(reviewProblems(task,source,label({state:'uncertain',bbox:null}))).toEqual([])});
 it('counts reviewer characters consistently with server validation',()=>{expect(reviewProblems(task,source,label({reviewer_id:'😀'}))).toHaveLength(1);expect(reviewProblems(task,source,label({reviewer_id:'😀A'}))).toEqual([])});
 it('requires reviewer and explicit source attestation',()=>{expect(reviewProblems(task,source,label({reviewer_id:'A',reviewed_from_source:false}))).toHaveLength(2)});
 it('prevents a pending release from being silently dropped on save',()=>{expect(reviewProblems({...task,kind:'passes',start_frame:10,end_frame:30},source,label({state:'reviewed',bbox:null}),15)[0]).toContain('Finish the pass starting at frame 15')});
 it('rejects same-frame, reversed and out-of-window pass endpoints',()=>{for(const [a,b] of [[15,15],[16,15],[9,20],[15,31]]){expect(reviewProblems({...task,kind:'passes',start_frame:10,end_frame:30},source,label({state:'reviewed',bbox:null,passes:[{release_frame:a,receipt_frame:b,outcome:'completed',sender_description:'',receiver_description:''}]}))).toHaveLength(1)}});
 it('accepts a fully reviewed no-pass window without inventing passes',()=>{expect(reviewProblems({...task,kind:'passes',start_frame:10,end_frame:30},source,label({state:'reviewed',bbox:null,exhaustive:true}))).toEqual([])});
 it('respects schema limits for reviewer IDs and pass counts',()=>{expect(reviewProblems(task,source,label({reviewer_id:'A'.repeat(81)}))[0]).toContain('80');const pass={release_frame:15,receipt_frame:20,outcome:'completed' as const,sender_description:'',receiver_description:''};expect(reviewProblems({...task,kind:'passes',start_frame:10,end_frame:30},source,label({state:'reviewed',bbox:null,passes:Array(31).fill(pass)}))[0]).toContain('30')});
});
describe('review frame clock and server feedback',()=>{
 it('does not mark frame 30 after seeking to the truncated start time of frame 31',()=>{expect(sourceReviewFrame(1.292958,23.976023976023978,541)).toBe(31)});
 it('keeps genuinely earlier times in the preceding frame',()=>{expect(sourceReviewFrame(31/23.976023976023978-.001,23.976023976023978,541)).toBe(30)});
 it('rejects unavailable clocks and bounds the final source frame',()=>{expect(sourceReviewFrame(NaN,25,100)).toBeNull();expect(sourceReviewFrame(1,0,100)).toBeNull();expect(sourceReviewFrame(10,25,100)).toBe(99)});
 it('shows backend string details instead of hiding the reason',()=>{expect(reviewServerError({detail:'Draw the two players'},422)).toBe('Draw the two players')});
 it('formats Pydantic field errors without losing their reason',()=>{expect(reviewServerError({detail:[{loc:['body','reviewer_id'],msg:'Too long'}]},422)).toBe('Reviewer ID: Too long')});
 it('keeps a useful fallback when a proxy returns a non-JSON error',()=>{expect(reviewServerError(null,502)).toContain('not saved (502)')});
});
