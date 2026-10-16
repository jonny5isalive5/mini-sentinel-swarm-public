"""Verify this publication's local bytes. No model execution, network or credentials."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
manifest=read('PUBLICATION_MANIFEST.json')['files']
actual={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and p.name!='PUBLICATION_MANIFEST.json'}
assert actual==manifest,'Publication file set or checksum differs'
provenance=read('PUBLICATION_PROVENANCE.json')
by_original={e['original_path']:e for e in provenance['source_snapshot_files']}
for e in by_original.values():
 if e['publication_path'] is not None:
  assert actual[e['publication_path']]==e['publication_sha256']
for path in provenance['protected_unchanged_paths']:
 e=by_original[path]
 assert e['modified'] is False and actual[path]==e['original_sha256']
archives={}
for e in provenance['archive_members']:
 path,member=e['path'].split('!/',1)
 if path not in archives:
  archives[path]=zipfile.ZipFile(ROOT/path)
  assert archives[path].testzip() is None
 assert sha(archives[path].read(member))==e['publication_sha256']
for path,z in archives.items():
 expected={e['path'].split('!/',1)[1] for e in provenance['archive_members'] if e['path'].startswith(path+'!/')}
 assert set(z.namelist())==expected
 z.close()
print(json.dumps({'status':'PASS','publication_files':len(actual),'protected_unchanged_files':len(provenance['protected_unchanged_paths']),'archive_members':len(provenance['archive_members']),'experiments_executed':0},indent=2))
