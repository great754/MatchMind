import {TemporalDiagnostics,temporalSchema} from './TemporalDebug';
import {z} from 'zod';
const localization=z.object({status:z.enum(['DIRECT','TRACKED','TEMPORAL_TRACKED','TEMPORAL_PREDICTED','TEMPORAL_CORRECTED','BIDIRECTIONAL_INTERPOLATED','PARTIAL_CONSTRAINED','RELOCALIZED','UNKNOWN']),outcome:z.string(),absolute_outcome:z.string(),tracking_blockers:z.array(z.string()),semantic_classes:z.array(z.string()),transition:z.string().nullable(),quality:z.string(),last_absolute_age_seconds:z.number().nullable(),camera_epoch:z.number(),support_score:z.number().nullable().optional(),current_semantic_support:z.boolean().optional(),relative_motion_error_budget_px:z.number().nullable().optional()});
export function CameraDebug({motion,region}:{motion:Record<string,unknown>,region?:Array<[number,number]>|null}){
 const parsed=localization.safeParse(motion.localization);const info=parsed.success?parsed.data:null;
 const temporal=temporalSchema.safeParse(motion.temporal_camera);
 const flow=z.object({matches:z.number(),inlier_ratio:z.number().optional(),heldout_median_small_px:z.number().optional(),heldout_p90_small_px:z.number().optional(),scale_to_source:z.number().optional()}).safeParse(temporal.success?temporal.data.field_motion:motion.field_motion??motion.projective_motion);
 return <section className="card cv-detail" aria-label="Camera State" data-testid="cv-camera-debug" data-status={info?.status??'UNKNOWN'}><h2>Camera State</h2>
 <p><strong>{info?.status??'UNKNOWN'}</strong> · {info?.quality??'No current camera evidence'}</p>
 <p>Outcome: {info?.outcome??'Unknown'} · Absolute fit: {info?.absolute_outcome??'Unknown'}</p>
 <p>Last absolute calibration: {info?.last_absolute_age_seconds==null?'Unknown':info.last_absolute_age_seconds.toFixed(3)+'s ago'} · Camera epoch: {info?.camera_epoch??'Unknown'}</p>
 <p>Transform support: {info?.support_score?.toFixed(3)??'Unknown'} (heuristic) · Current semantic support: {info?.current_semantic_support==null?'Unknown':info.current_semantic_support?'Yes':'No'} · Relative image-motion budget: {info?.relative_motion_error_budget_px?.toFixed(3)??'Unknown'} source pixels</p>
 <p>Visible pitch region: {region?.length?`x=${Math.min(...region.map(p=>p[0])).toFixed(1)}–${Math.max(...region.map(p=>p[0])).toFixed(1)} m · y=${Math.min(...region.map(p=>p[1])).toFixed(1)}–${Math.max(...region.map(p=>p[1])).toFixed(1)} m`:"Unknown"}</p>
 <p>Pitch evidence: {info?.semantic_classes.length??'Unknown'} semantic classes ({info?.semantic_classes.join(', ')||'Unknown'}) · Transition: {info?.transition??'None'}</p>
 {info?.status==='UNKNOWN'&&<p>Localization unavailable: {info.tracking_blockers.join(' · ')||info.absolute_outcome}. Current positions are Unknown.</p>}
 {flow.success&&<p>Motion matches: {flow.data.matches} · Inlier ratio: {flow.data.inlier_ratio?.toFixed(3)??'Unknown'} · Held-out motion error: {flow.data.heldout_median_small_px?.toFixed(3)??'Unknown'} median / {flow.data.heldout_p90_small_px?.toFixed(3)??'Unknown'} P90 pixels ({motion.field_motion?'640px working width; source scale '+(flow.data.scale_to_source??1).toFixed(3):'320 × 180 image'})</p>}
 <TemporalDiagnostics motion={motion}/>
 <p className="section-footnote">Direct fits and tracked geometry are heuristic. These checks do not measure independent camera accuracy. Historical ghosts are not current observations.</p></section>
}

const projection=z.object({state:z.string(),source:z.string(),epoch:z.number(),support:z.number().nullable(),last_direct_age_seconds:z.number().nullable(),current_semantic_support:z.boolean(),image_foot:z.tuple([z.number(),z.number()]).nullable().optional(),temporal_direction:z.string().nullable().optional(),anchor_age_seconds:z.number().nullable().optional(),forward_backward_agreement_p90_m:z.number().nullable().optional()});
export function CameraProjectionDetails({evidence}:{evidence:unknown}){
 const parsed=projection.safeParse(evidence);if(!parsed.success)return null;const p=parsed.data;
 return <div data-testid="player-camera-evidence"><p>Camera projection: {p.state} · {p.source} · epoch {p.epoch}</p><p>Last direct: {p.last_direct_age_seconds?.toFixed(3)??'Unknown'}s ago · Support: {p.support?.toFixed(3)??'Unknown'} (heuristic)</p><p>Current semantic support: {p.current_semantic_support?'Yes':'No'} · image_foot: {p.image_foot?.map(n=>n.toFixed(2)).join(', ')??'Unknown'} pixels (bounding-box bottom center)</p>{p.temporal_direction&&<p>Temporal direction: {p.temporal_direction} · Absolute anchor: {p.anchor_age_seconds==null?'Unknown':p.anchor_age_seconds<0?`future evidence in ${(-p.anchor_age_seconds).toFixed(3)}s`:`${p.anchor_age_seconds.toFixed(3)}s ago`} · Forward/backward agreement: {p.forward_backward_agreement_p90_m?.toFixed(3)??'Unavailable'} m P90</p>}</div>
}
