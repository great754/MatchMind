"""Separate sealed source-only review. No prediction, world model or canonical readiness required."""
import logging
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse
from src.serving.cv import digest
from src.serving.cv_review_labels import HumanReviewStore, review_router


class SourceOnlyReview:
    def __init__(self, metadata):
        self.manifest = metadata
        self.video = Path(metadata['source_video']).resolve()
        if digest(self.video) != metadata['video_sha256']:
            raise ValueError('Sealed review video changed')
        self.stamp = self.video.stat().st_mtime_ns

    def frame(self, timestamp):
        if self.video.stat().st_mtime_ns != self.stamp:
            raise HTTPException(503, 'Sealed review source changed; revalidate before reviewing')


def create_review_router(authorize, task_path='configs/evaluation/cv_m21_holdout.json', root='outputs/m21/human_review'):
    store = HumanReviewStore(task_path=task_path, root=root)
    sources = {}
    for name, metadata in store.dataset['sources'].items():
        try:
            sources[name] = SourceOnlyReview(metadata)
        except (OSError, ValueError):
            logging.getLogger('matchmind.requests').warning('M21 sealed review source unavailable: %s', name)
    router, store = review_router(authorize, sources, store=store, prefix='/cv/m21')

    @router.get('/review/sources')
    def source_list():
        rows = store.labels()
        return {'reference_role': 'sealed_final_test_pending_human_labels', 'source_match_groups': 1,
                'task_count': len(store.tasks), 'model_suggestions_shown': False,
                'sources': [{'recording_id': name, 'available': name in sources,
                    'tasks': sum(t['recording_id'] == name for t in store.tasks.values()),
                    'saved_tasks': len({r['task_id'] for r in rows if store.tasks[r['task_id']]['recording_id'] == name})}
                    for name in store.dataset['sources']]}

    @router.get('/matches/{recording}/video.mp4')
    def source_video(recording: str):
        source = sources.get(recording)
        if source is None:
            raise HTTPException(404, 'Sealed review source unavailable')
        source.frame(0)
        return FileResponse(source.video, media_type='video/mp4', headers={'Cache-Control':'private, max-age=60'})

    return router, store
