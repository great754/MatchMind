import {z} from 'zod';
const counts=z.object({tp:z.number(),fp:z.number(),fn:z.number(),precision:z.number().nullable(),recall:z.number().nullable(),f1:z.number().nullable()}).passthrough();
const measure=z.object({ball:counts,passes:counts}).passthrough();
export const footballMetadataSchema=z.object({recording_id:z.string(),reference_role:z.literal('development_only'),checkpoint_sha256:z.string(),
 model:z.object({architecture:z.string(),epoch:z.number(),split:z.object({train_sources:z.array(z.string()),validation_sources:z.array(z.string()),excluded_development_source:z.string()}).passthrough()}).passthrough(),
 metrics:measure,aggregate:measure,detector_only:z.object({ball:counts}).passthrough().nullable().optional(),baselines:z.record(z.string(),z.object({aggregate:measure,per_recording:z.record(z.string(),measure)})),
 completed_review:z.object({frames:z.number(),visible_frames:z.number(),systems:z.array(z.object({id:z.string(),ball:counts})),combined:counts}).optional(),
 promotion:z.object({status:z.string(),reason:z.string()}).passthrough(),independent_test_status:z.string(),
 training:z.object({verified_training_frames:z.number(),verified_validation_frames:z.number(),epochs_run:z.number(),weak_training_frames:z.number().optional()}).passthrough(),
 runtime:z.object({processing_fps:z.number(),inference_seconds:z.number(),peak_rss_bytes_macos:z.number()}).passthrough(),
 pass_trace:z.array(z.object({release_seconds:z.number(),receipt_seconds:z.number(),release_frame:z.number(),receipt_frame:z.number(),matched_completed_pass:z.boolean(),diagnostic_reasons:z.array(z.string())}).passthrough())});
export const footballFrameSchema=z.object({schema_version:z.literal('cv-m22-frame-v1'),recording_id:z.string(),frame:z.number(),timestamp:z.number(),camera_epoch:z.number(),context_frames:z.tuple([z.number(),z.number(),z.number()]),checkpoint_sha256:z.string(),ball_revision:z.string().optional(),directly_observed:z.boolean(),
 candidates:z.array(z.object({candidate_id:z.string(),bbox:z.tuple([z.number(),z.number(),z.number(),z.number()]),confidence:z.number(),current_only_confidence:z.number(),observation_eligible:z.boolean(),decision_reason:z.string(),selected:z.boolean()})),heatmap:z.array(z.array(z.number())),heatmap_is_calibrated:z.literal(false),
 association:z.literal('existing_m21_offline_dag'),uses_future_for_association:z.literal(true),reconstruction:z.object({image:z.tuple([z.number(),z.number()]),support_frames:z.tuple([z.number(),z.number()])}).passthrough().nullable(),
 interaction:z.object({state:z.string(),reason:z.string(),possible_player_entity:z.string().nullable(),verified_touch:z.literal(false),verified_possession:z.literal(false)}).passthrough()});
export type FootballFrame=z.infer<typeof footballFrameSchema>;
