export const DEFAULT_EVENT_PREROLL_SECONDS: number;
export type PlaybackEvent={timestamp:number;source:string;metrics?:Record<string,unknown>;tactical?:{evidence_start_frame?:number|null}|null};
export function getPlaybackAnchorTime(timestamp:number,event:PlaybackEvent|null|undefined,fps:number):number;
export function getPlaybackSeekTime(timestamp:number,event:PlaybackEvent|null|undefined,fps:number,duration:number,preRoll?:number):number;
