"""Explicit annotation evidence contracts: missing boxes do not establish negative visibility."""
import math

def validate_annotation(row):
 if row['width']<=0 or row['height']<=0 or row['timestamp']<0:raise ValueError('Invalid source dimensions/clock')
 for b in row['balls']:
  x,y,w,h=b['bbox']
  if not all(math.isfinite(v) for v in b['bbox']) or min(x,y)<0 or min(w,h)<=0 or x+w>row['width'] or y+h>row['height']:raise ValueError('Invalid ball box')
 if row.get('verified_negative') and row['balls']:raise ValueError('Negative visibility contradicts ball boxes')
 if row.get('visibility')=='VISIBLE' and not row['balls']:raise ValueError('Visible ball requires a box')
 if row.get('verified_negative') and (row.get('visibility')!='NO_VISIBLE_BALL' or not row.get('independent_review')):raise ValueError('Explicit independent negative review required')
 if row.get('metric_reference') and (row.get('height')!='GROUND_VERIFIED' or not row.get('independent_metric_provenance')):raise ValueError('Independent ground/metric reference required')
 return row

def check_splits(rows,historical_test=()):
 seen={};games={};old=set(historical_test)
 for r in rows:
  validate_annotation(r);s=r['split'];match=r['match_id'];h=r['source_sha256']
  if match in games and games[match]!=s:raise ValueError('Whole-match split leakage')
  if h in seen and seen[h]!=s:raise ValueError('Source duplicate across splits')
  if s=='train' and r['sample_id'] in old:raise ValueError('Historical TEST in training')
  seen[h]=s;games[match]=s
 return True
