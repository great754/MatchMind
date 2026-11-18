export type ReviewBox=[number,number,number,number];
export type ReviewTask={task_id:string,kind:'ball'|'identity'|'passes',frame?:number,other_frame?:number,start_frame?:number,end_frame?:number,start?:number,end?:number};
export type ReviewSource={fps:number,width:number,height:number,frames:number};
export type ReferencePass={release_frame:number,receipt_frame:number,outcome:'completed'|'uncertain',sender_description:string,receiver_description:string};
export type ReviewLabel={task_id:string,reviewer_id:string,origin:'human',reviewed_from_source:boolean,model_assisted:false,state:string,bbox:ReviewBox|null,other_bbox:ReviewBox|null,exhaustive:boolean,passes:ReferencePass[],notes:string};

export function sourceReviewFrame(time:number,fps:number,frames:number):number|null{
 if(!Number.isFinite(time)||time<0||!Number.isFinite(fps)||fps<=0||!Number.isInteger(frames)||frames<1)return null;
 // Browser media times truncate to microseconds. Allow one microsecond at a
 // frame boundary, so a seek to frame 31 cannot accidentally mark frame 30.
 return Math.min(frames-1,Math.floor((time+0.000001)*fps));
}

export function validReviewBox(box:ReviewBox|null,source:ReviewSource):boolean{
 if(!box||!box.every(Number.isFinite))return false;
 const [x,y,width,height]=box;
 return x>=0&&y>=0&&width>0&&height>0&&x+width<=source.width&&y+height<=source.height;
}

export function reviewProblems(task:ReviewTask,source:ReviewSource,label:ReviewLabel,pendingRelease:number|null=null):string[]{
 const problems:string[]=[];
 if(Array.from(label.reviewer_id.trim()).length<2)problems.push('Enter a reviewer ID with at least two characters.');
 if(Array.from(label.reviewer_id).length>80)problems.push('Keep the reviewer ID within 80 characters.');
 if(!label.reviewed_from_source)problems.push('Confirm that you personally reviewed the original source.');
 if(Array.from(label.notes).length>500)problems.push('Keep notes within 500 characters.');
 const checkBox=(box:ReviewBox|null,name:string,required:boolean)=>{
  if(!box&&required)problems.push(`Draw a box around ${name}.`);
  else if(box&&!validReviewBox(box,source))problems.push(`Redraw the box for ${name} with positive width and height inside the image, or clear it.`);
 };
 if(task.kind==='ball'){
  if(!['visible','not_visible','uncertain'].includes(label.state))problems.push('Choose a ball visibility result.');
  checkBox(label.bbox,'the visible ball',label.state==='visible');
  if(label.state!=='visible'&&label.bbox)problems.push('Hidden or uncertain balls must be saved without a box.');
 }else if(task.kind==='identity'){
  if(!['same','different','uncertain'].includes(label.state))problems.push('Choose whether the players are the same, different or uncertain.');
  checkBox(label.bbox,'the player in the first image',label.state!=='uncertain');
  checkBox(label.other_bbox,'the compared player in the second image',label.state!=='uncertain');
 }else{
  if(!['reviewed','uncertain'].includes(label.state))problems.push('Choose Reviewed or Uncertain for this window.');
  if(pendingRelease!==null)problems.push(`Finish the pass starting at frame ${pendingRelease} by marking a later receipt, or cancel the release mark.`);
  if(label.passes.length>30)problems.push('A review window supports up to 30 pass references.');
  for(const [index,pass] of label.passes.entries()){
   if(!Number.isInteger(pass.release_frame)||!Number.isInteger(pass.receipt_frame)||task.start_frame===undefined||task.end_frame===undefined||pass.release_frame<task.start_frame||pass.receipt_frame>task.end_frame||pass.release_frame>=pass.receipt_frame){
    problems.push(`Pass ${index+1} needs a release before its receipt, both inside frames ${task.start_frame}–${task.end_frame}.`);
   }
  }
 }
 return problems;
}

export function reviewServerError(body:unknown,status:number):string{
 const detail=body&&typeof body==='object'&&'detail' in body?body.detail:null;
 if(typeof detail==='string')return detail;
 if(Array.isArray(detail)){
  const names:Record<string,string>={reviewer_id:'Reviewer ID',reviewed_from_source:'Source review confirmation',bbox:'First box',other_bbox:'Second box',passes:'Pass reference',notes:'Notes'};
  const messages=detail.map(error=>{
   if(!error||typeof error!=='object'||typeof error.msg!=='string')return '';
   const location=Array.isArray(error.loc)?error.loc.filter((v:unknown)=>v!=='body'):[];
   const field=names[String(location[0])]??location.map(String).join(' ');
   return `${field?field+': ':''}${error.msg}`;
  }).filter(Boolean);
  if(messages.length)return messages.join(' ');
 }
 return `The review was not saved (${status}). Please try again; your edits are still here.`;
}
