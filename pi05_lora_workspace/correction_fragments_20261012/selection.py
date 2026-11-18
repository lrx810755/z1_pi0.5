"""Require an explicit frozen task selection when collection is stopped early."""
def check_requested_tasks(plan_ids,completed,requested):
    if len(set(requested))!=len(requested) or not set(requested)<=set(plan_ids):
        raise ValueError('Snapshot contains duplicate or unknown tasks')
    if set(requested)!=set(completed):
        raise ValueError('Requested snapshot differs from accepted completed tasks')
    if not requested:
        raise ValueError('No accepted tasks selected')
    return len(requested)
