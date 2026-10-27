import {z} from 'zod';
import {agentSchema,answerSchema,viewSchema,factSchema} from './contracts';
import type {Mode,Provider} from './contracts';

export class ApiError extends Error {constructor(public status:number,message:string){super(message)}}
export async function request<T>(path:string,schema:z.ZodType<T>,options:RequestInit={},timeout=80000):Promise<T>{
 const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),timeout);
 try {const response=await fetch(path,{...options,signal:controller.signal,headers:{'Content-Type':'application/json',...options.headers}});
 if(!response.ok)throw new ApiError(response.status,response.status===404?'This recording or evidence is unavailable.':response.status===401?'The service requires authorization.':`The service could not complete this request (${response.status}).`);
 const result=schema.safeParse(await response.json());if(!result.success)throw new Error('The service returned an unexpected data format.');return result.data;
 }catch(e){if(e instanceof ApiError)throw e;if(controller.signal.aborted)throw new Error('The request timed out. Try again or choose Local.');if(e instanceof TypeError)throw new Error('Cannot reach the API. Check that the local service is running.');throw e;}finally{clearTimeout(timer)}
}
export function apiForRecording(recordingId:string){
 if(!/^[A-Za-z0-9_-]{1,80}$/.test(recordingId))throw new Error('Invalid recording');
 const base='/api/matches/'+encodeURIComponent(recordingId);
 return {view:()=>request(base+'/view',viewSchema,{},20000),ask:(question:string,mode:Mode,provider:Provider,timestamp:number,conversation:Record<string,unknown>)=>request(base+'/ask',agentSchema,{method:'POST',body:JSON.stringify({question,mode,provider,timestamp,conversation})}),explain:(id:string,mode:Mode)=>request(base+'/events/'+encodeURIComponent(id)+'/explain?mode='+mode+'&with_evidence=true',agentSchema),player:(id:number)=>request(base+'/players/'+id,z.object({status:z.string(),facts:z.array(factSchema),limitations:z.array(z.string())}))};
}
export const api=apiForRecording('118575_clip');
export const recordingList=()=>request('/api/recordings',z.object({recordings:z.array(z.object({recording_id:z.string(),match_id:z.string(),status:z.enum(['available','not_processed','unavailable','error']),media_available:z.boolean().optional(),media_url:z.string().nullable().optional(),source:z.string().optional(),capabilities:z.object({cv_world_model_available:z.boolean(),analytics_available:z.boolean(),ask_available:z.boolean()}).passthrough().optional()}))}));
export const mediaUrl=(path:string)=>'/api'+path;
