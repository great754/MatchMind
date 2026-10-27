import unittest
from pathlib import Path
from src.serving.playback import report_playback

class PlaybackReportTests(unittest.TestCase):
    def test_in_memory_report_adaptation_does_not_mutate_evidence(self):
        original='<html><script>function seek(t){};seek(event.timestamp);seek(statement.timestamp);seek(askContext.last_timestamp);slider.addEventListener("input",()=>seek(Number(slider.value)));</script></html>'
        events=[dict(timestamp=12,source='transfer',metrics=dict(start_frame=200))]
        page=report_playback(original,events,25)
        self.assertIn('seekMoment(event.timestamp,event.event_id)',page)
        self.assertIn('seekMoment(statement.timestamp,statement.event_id)',page)
        self.assertIn('seekMoment(askContext.last_timestamp,askContext.last_event_id)',page)
        self.assertIn('()=>seek(Number(slider.value))',page)
        self.assertEqual(events[0]['timestamp'],12)
        self.assertEqual(report_playback(page,events,25),page)
    def test_report_escapes_script_content(self):
        page=report_playback('<script>seek(item.timestamp)</script>',[{'event_id':'</script>','timestamp':10}],25)
        self.assertIn(r'\u003c/script>',page)
        self.assertIn('seekMoment(item.timestamp,item.event_ids?.[0])',page)
