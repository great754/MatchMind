import {render,screen} from '@testing-library/react';
import {describe,it,expect} from 'vitest';
import {PersistentFieldDiagnostics} from './TemporalDebug';
describe('persistent camera evidence',()=>{
 it('keeps missing evidence unavailable',()=>{render(<PersistentFieldDiagnostics evidence={{}}/>);expect(screen.queryByTestId('persistent-field-diagnostics')).toBeNull()});
 it('explains fixed-world inherited anchors and rejected fits',()=>{render(<PersistentFieldDiagnostics evidence={{world_field:{active_world_anchors:12,accepted:false,reason:'INSUFFICIENT_WORLD_FIELD_ANCHORS'},trajectory_uncertainty_px:1.2,last_accepted_frame:25,new_world_anchors:0,feature_survival_fraction:.5,m7_trajectory_review:{forward_valid:false,backward_valid:true,accepted:true}}}/>);expect(screen.getByTestId('persistent-field-diagnostics')).toHaveTextContent('World-anchored features: 12');expect(screen.getByTestId('persistent-field-diagnostics')).toHaveTextContent('Backward estimate: Accepted');expect(screen.getByTestId('persistent-field-diagnostics')).toHaveTextContent('not independently surveyed')});
});
