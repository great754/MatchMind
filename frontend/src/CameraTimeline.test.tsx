import {describe,it,expect,vi} from 'vitest';
import {render,screen,fireEvent,cleanup} from '@testing-library/react';
import {CameraTimeline} from './CameraTimeline';
describe('camera timeline',()=>{
 it('shows frame evidence and seeks the exact diagnostic timestamp',()=>{
  const onSeek=vi.fn();const ticks=[{frame:0,timestamp:0,state:'UNKNOWN',reason:'NO_PRIOR',last_direct_age_seconds:null,support:null,landmarks:[]},{frame:1,timestamp:.041708333,state:'PARTIAL_CONSTRAINED',reason:'CURRENT_LANDMARKS_AGREE',last_direct_age_seconds:.1,support:.8,landmarks:['Circle central','Side line top']}];
  render(<CameraTimeline ticks={ticks} frame={1} onSeek={onSeek}/>);fireEvent.click(screen.getByRole('button',{name:'Inspect camera frame 1: PARTIAL_CONSTRAINED'}));expect(onSeek).toHaveBeenLastCalledWith(.041708333);expect(screen.getByTestId('camera-timeline-selection').textContent).toContain('CURRENT_LANDMARKS_AGREE');expect(screen.getByText(/Direct age: 0.100s/).textContent).toContain('Circle central');fireEvent.keyDown(screen.getByRole('button',{name:'Inspect camera frame 0: UNKNOWN'}),{key:'Enter'});expect(onSeek).toHaveBeenLastCalledWith(0);cleanup();
 });
});
