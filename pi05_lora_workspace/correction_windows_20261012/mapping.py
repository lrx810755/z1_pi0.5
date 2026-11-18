"""Resolve source-local correction windows into actual dataset row indices."""
def map_records(records,source_indices,episode_bounds):
    mapped=[]
    for record in records:
        r=dict(record)
        if r['source']=='original':index=r['episode_index']
        else:
            if r['source_path'] not in source_indices:raise ValueError('Unmapped source identity')
            index=source_indices[r['source_path']]
        if index not in episode_bounds:raise ValueError('Missing actual episode bounds')
        start,length=episode_bounds[index]
        if length!=r['frame_count']:raise ValueError('Recorded and converted frame counts differ')
        if not (0<=r['anchor_start']<=r['anchor_end_inclusive']<length):raise ValueError('Invalid local interval')
        r.update(episode_index=index,episode_id_scope='actual_merged_211',dataset_absolute_start=start,
                 absolute_anchor_start=start+r['anchor_start'],absolute_anchor_end_inclusive=start+r['anchor_end_inclusive'])
        mapped.append(r)
    if len({r['episode_index'] for r in mapped})!=len(mapped):raise ValueError('Duplicate actual episode identity')
    return sorted(mapped,key=lambda r:r['episode_index'])
