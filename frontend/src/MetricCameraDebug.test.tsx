import {render,screen,cleanup} from '@testing-library/react';
import {describe,it,expect} from 'vitest';
import {MetricCameraDebug,MetricConicOverlay,MetricCameraPath} from './MetricCameraDebug';
const evidence={accepted:false,state:'UNKNOWN',reason:'INSUFFICIENT_CURRENT_LANDMARKS',placement_provenance:'UNKNOWN',current_transform:null,incremental_view_shift_m:null,accumulated_view_shift_m:null,history:[],landmark_certificate:{kind:'CENTER_CIRCLE',accepted:false,reason:'INSUFFICIENT_CURRENT_LANDMARKS',conic:{accepted:true,reason:'BROAD_HELDOUT_CONIC',ellipse_center_image:[400,300],ellipse_axes:[200,80],ellipse_angle_degrees:2,arc_span_degrees:160},candidates:[]}};
describe('metric camera evidence',()=>{
 it('does not mistake an observed conic for an accepted metric camera',()=>{render(<MetricCameraDebug motion={{metric_camera:evidence}}/>);expect(screen.getByTestId('metric-camera-debug')).toHaveTextContent('Current camera: Unknown');expect(screen.getByTestId('metric-landmark-decision')).toHaveTextContent('Rejected/unavailable');expect(screen.getByTestId('metric-camera-debug')).toHaveTextContent('not the physical 3D camera position');cleanup()});
 it('renders the observed conic separately from canonical pitch geometry',()=>{render(<svg><MetricConicOverlay motion={{metric_camera:evidence}}/></svg>);expect(screen.getByTestId('observed-metric-conic')).toHaveAttribute('rx','100');cleanup()});
 it('never draws a camera path through an unknown segment',()=>{render(<svg><MetricCameraPath motion={{metric_camera:evidence}}/></svg>);expect(screen.queryByTestId('metric-camera-path')).toBeNull();cleanup()});
});
