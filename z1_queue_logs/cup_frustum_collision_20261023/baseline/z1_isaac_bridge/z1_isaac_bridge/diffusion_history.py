"""Read diffusion history at the training interval, independent of chunk execution length."""
from .recording_sync import RecordingSynchronizer

def previous_observation(synchronizer,current,interval=.1):
    reader=RecordingSynchronizer(synchronizer.cameras,tolerance=synchronizer.tolerance,
                                state_gap=synchronizer.state_gap)
    reader.buffers=synchronizer.buffers
    previous=reader.select_observation(current['time']-interval)
    for camera in synchronizer.cameras:
        gap=current['images'][camera][0]-previous['images'][camera][0]
        if not interval-synchronizer.tolerance-1e-8<=gap<=interval+synchronizer.tolerance+1e-8:
            raise ValueError(f'{camera}: invalid diffusion history interval {gap}')
    return previous
