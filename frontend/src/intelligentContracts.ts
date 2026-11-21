import {z} from 'zod';
const n=z.number().finite(), point=z.tuple([n,n]), box=z.tuple([n,n,n.positive(),n.positive()]);
const covariance=z.tuple([point,point]).refine(c=>c[0][0]>=0&&c[1][1]>=0&&Math.abs(c[0][1]-c[1][0])<1e-5&&c[0][0]*c[1][1]>=c[0][1]**2-1e-5,'Invalid covariance');
const sha=z.string().regex(/^[a-f0-9]{64}$/);
export const intelligentFrameSchema=z.object({
 schema_version:z.literal('cv-m23-frame-v1'),recording_id:z.string(),variant:z.string(),frame:n.int().nonnegative(),timestamp:n.nonnegative(),camera_epoch:n.int(),
 state:z.enum(['OBSERVED','PREDICTED','AMBIGUOUS','UNKNOWN']),directly_observed:z.boolean(),selected_candidate_id:z.string().nullable(),
 candidates:z.array(z.object({candidate_id:z.string(),bbox:box,confidence:n.min(0).max(1),origin:z.enum(['primary','guided','discovery','complement']),visual_evidence:z.literal('CURRENT_PIXELS'),observation_eligible:z.boolean(),selected:z.boolean(),decision_reason:z.string(),blur_supported:z.boolean(),current_only_confidence:n.nullable().optional()})),
 memory:z.object({state:z.string(),missed_frames:n.nullable(),missed_seconds:n.nullable(),last_direct_observation:z.object({position:point,frame:n,timestamp:n}).nullable(),hypotheses:z.array(z.object({mean:point,covariance,velocity:point,acceleration:point,track_id:n,confirmed:z.boolean(),archived:z.boolean(),age_seconds:n})),search_regions:z.array(z.object({bbox:box,track_id:n,kind:z.string().optional()})),camera_motion_source:z.string(),camera_cut:z.boolean(),reacquired_with_current_visual_evidence:z.boolean(),predicted_image_position:point.nullable(),reason:z.string()}).nullable(),
 reconstruction:z.object({image:point,directly_observed:z.literal(false),support_frames:z.array(n),uses_future:z.literal(true),reason:z.string()}).passthrough().nullable(),
 trajectory_history:z.array(z.object({frame:n,current_camera_image:point,current_observation:z.literal(false)})),
 interaction:z.object({state:z.string(),reason:z.string(),verified_touch:z.literal(false),verified_possession:z.literal(false)}).passthrough(),
 reasons:z.array(z.string()),source_pixels_sha256:sha,model_set_sha256:sha,config_sha256:sha,ball_revision:sha.optional(),uses_future_for_association:z.literal(true),future_visual_frames_used:z.literal(false),
}).superRefine((r,c)=>{const selected=r.candidates.filter(x=>x.selected);if(r.directly_observed!==(r.state==='OBSERVED')||selected.length!==(r.directly_observed?1:0)||selected.some(x=>!x.observation_eligible||x.confidence<.35||x.candidate_id!==r.selected_candidate_id||x.origin==='complement'&&(x.current_only_confidence??0)<.35))c.addIssue({code:'custom',message:'M23 current visual observation provenance mismatch'});if(r.reconstruction&&r.directly_observed)c.addIssue({code:'custom',message:'Reconstruction is not an observation'});});
export type IntelligentFrame=z.infer<typeof intelligentFrameSchema>;
export const detectionMetrics=z.object({tp:n,fp:n,fn:n,precision:n.nullable(),recall:n.nullable(),f1:n.nullable()});
const nav=z.object({kind:z.string(),frame:n,receipt_frame:n.optional()}).passthrough();
export const intelligentMetadataSchema=z.object({recording_id:z.string(),model_set_sha256:sha,ball_revision:sha,reference_role:z.enum(['development_only','unlabelled_inference','reviewed_development']),promotion:z.object({status:z.string(),ordinary_default:z.literal('m14'),reason:z.string()}),independent_final_test:z.string(),
 completed_review:z.object({completed_tasks:n.int().nonnegative(),visible_frames:n.int().nonnegative(),uncertain_frames:n.int().nonnegative(),absent_frames:n.int().nonnegative(),rejected_regions:n.int().nonnegative(),
  systems:z.array(z.object({id:z.string(),ball:detectionMetrics})),failure_counts:z.record(z.string(),n),evaluated_at_utc:z.string(),annotation_snapshot_sha256:sha,labels_updated_since_snapshot:z.boolean(),reference_role:z.literal('development_only'),model_assisted:z.literal(true),independent_final_test:z.literal(false)}).nullable().optional(),
 completed_review_diagnostic:z.string().nullable().optional(),
 metrics:z.object({navigation:z.array(nav),engineering:z.object({states:z.record(z.string(),n)}),pass_traces:z.array(z.object({release_frame:n,receipt_frame:n,matched:z.boolean().optional()}).passthrough())}),
 comparisons:z.array(z.object({id:z.string(),new_150:detectionMetrics,original:detectionMetrics,passes:detectionMetrics})),
 frozen_baselines:z.array(z.object({id:z.string(),new_150:detectionMetrics})),
 runtime:z.object({processing_fps:n,recorded_processing_seconds:n,timing:z.object({extra_crop_views:n}),peak_rss_bytes_macos:n}),
});
