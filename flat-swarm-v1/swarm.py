#!/usr/bin/env python3
"""swarm.py -- the Mini Sentinel specialist swarm: a thin coordination layer over the RELEASED uai32 v1.0.0
binary (../uai32/dist/uai32, 9,188 bytes, SHA-256 996d73cb...).

Every class prediction and every weight update in this project is made by that binary on a model file.  What
this file does in numpy is fit 21 unsupervised statistics per specialist (centroid, spread, radius of its training
inputs) and apply fixed formulas (Gaussian familiarity score, vote weight, 0.75 disagreement margin, composition
table, 0.1 spawn-cluster threshold): a label-free nearest-centroid router, not a classifier of the task.  It also
generates the synthetic tasks (the test generator), keeps the library, and measures.  No larger model is involved
at build time or run time.

Brain = one uai32 model file (10 inputs, 16 hidden, 4 classes = 576 bytes).  A specialist is a brain plus a JSON
metadata record.  Specialists are never resident: each invocation is a fresh uai32 process that loads one file.
"""
import argparse, hashlib, io, json, os, resource, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UAI32 = os.environ.get('UAI32', os.path.join(HERE, '..', 'uai32', 'dist', 'uai32'))
NI_ID, NI_COND, NO, NH = 4, 6, 4, 16           # identifying features, condition features, classes, hidden units
NI = NI_ID + NI_COND
CONDITIONS = ['normal', 'noisy', 'shifted', 'edge', 'adversarial']
WORK = os.path.join(HERE, 'work'); os.makedirs(WORK, exist_ok=True)
_tmp = 0

def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

def rows_text(X, y=None):
    """uai32's data format: one example per line, features then the integer label."""
    buf = io.StringIO()
    if y is None: np.savetxt(buf, np.asarray(X), fmt='%.5f')
    else: np.savetxt(buf, np.column_stack([X, y]), fmt=' '.join(['%.5f'] * X.shape[1] + ['%d']))
    return buf.getvalue()

def tmpfile(text):
    global _tmp; _tmp += 1
    p = os.path.join(WORK, f'rows_{os.getpid()}_{_tmp}.txt'); open(p, 'w').write(text); return p

class Sentinel:
    """Wrapper over the uai32 binary.  Each call is a fresh process; the process exits when done, so nothing stays
    resident.  Counters give the invocation cost."""
    calls = 0; seconds = 0.0; failures = 0
    @staticmethod
    def run(args, stdin=None):
        t = time.perf_counter()
        p = subprocess.run([UAI32] + args, input=stdin, capture_output=True, text=True)
        Sentinel.seconds += time.perf_counter() - t; Sentinel.calls += 1
        for a in args:                                        # temporary row files are deleted once consumed
            if a.startswith(WORK + '/rows_') and os.path.exists(a): os.remove(a)
        if p.returncode: Sentinel.failures += 1; raise RuntimeError(f'uai32 {" ".join(args[:2])} failed: {p.stderr.strip()}')
        return p.stdout
    @staticmethod
    def create(model, X, y, nh=NH, seed=1):          # new brain: normalisation from X, random He-init weights, 0 epochs
        if os.path.exists(model): os.remove(model)    # uai32 would otherwise CONTINUE training an existing file
        Sentinel.run(['train', tmpfile(rows_text(X, y)), model, str(nh), '0', '0.05', str(seed)])
    @staticmethod
    def train(model, X, y, epochs, lr=0.05, seed=1):  # continue training an EXISTING model file (HIDDEN ignored = 0)
        Sentinel.run(['train', tmpfile(rows_text(X, y)), model, '0', str(epochs), str(lr), str(seed)])
    @staticmethod
    def predict(model, X):
        out = Sentinel.run(['predict', model], stdin=rows_text(X))
        rows = [l.split() for l in out.strip().split('\n')]
        return np.array([int(r[0]) for r in rows]), np.array([max(map(float, r[1:])) for r in rows])
    @staticmethod
    def accuracy(model, X, y):
        out = Sentinel.run(['test', tmpfile(rows_text(X, y)), model])
        k, n = out.split('accuracy')[1].split('=')[0].strip().split('/')
        return int(k), int(n)

class World:
    """Seeded synthetic task family (the 'test generator').  Domain d: 4 identifying features ~ N(c_d, 0.03) with
    centres c_d in [-1,1]^4 at least 0.15 apart; 6 condition features u ~ U(-1,1); label = argmax of a random
    2-layer teacher t_d(u) (6 -> 8 tanh -> 4), re-drawn until every class has >= 10% mass.  Conditions change the
    OBSERVED u (labels always come from the clean u): noisy (+N(0,0.3)), shifted (0.7u+0.2), edge (smallest teacher
    margin), adversarial (perturbation of norm 0.2 chosen to favour a wrong class)."""
    def __init__(self, n_domains, seed=0):
        self.n, self.seed = n_domains, seed
        rng = np.random.default_rng(seed)
        self.centres = np.zeros((n_domains, NI_ID)); placed = 0; tries = 0
        while placed < n_domains:
            c = rng.uniform(-1, 1, NI_ID); tries += 1
            if placed == 0 or np.min(np.linalg.norm(self.centres[:placed] - c, axis=1)) > 0.15:
                self.centres[placed] = c; placed += 1
            if tries > 200000: raise RuntimeError('could not place domain centres')
        self.W1, self.b1, self.W2, self.b2 = [], [], [], []
        probe = rng.uniform(-1, 1, (500, NI_COND))
        for d in range(n_domains):
            while True:
                W1, b1 = rng.normal(0, 1.5, (NI_COND, 8)), rng.normal(0, 0.5, 8)
                W2, b2 = rng.normal(0, 1.5, (8, NO)), rng.normal(0, 0.5, NO)
                lab = np.argmax(np.tanh(probe @ W1 + b1) @ W2 + b2, axis=1)
                if np.bincount(lab, minlength=NO).min() >= 50: break
            self.W1.append(W1); self.b1.append(b1); self.W2.append(W2); self.b2.append(b2)
    def logits(self, d, u): return np.tanh(u @ self.W1[d] + self.b1[d]) @ self.W2[d] + self.b2[d]
    def labels(self, d, u): return np.argmax(self.logits(d, u), axis=1)
    def sample(self, d, n, rng, condition='normal'):
        if condition == 'edge':
            cand = rng.uniform(-1, 1, (n * 10, NI_COND)); z = np.sort(self.logits(d, cand), axis=1)
            u = cand[np.argsort(z[:, -1] - z[:, -2])[:n]]
        else:
            u = rng.uniform(-1, 1, (n, NI_COND))
        y = self.labels(d, u); obs = u.copy()
        if condition == 'noisy': obs = u + rng.normal(0, 0.3, u.shape)
        elif condition == 'shifted': obs = 0.7 * u + 0.2
        elif condition == 'adversarial':
            for i in range(n):
                P = rng.normal(size=(8, NI_COND)); P *= 0.2 / np.linalg.norm(P, axis=1, keepdims=True)
                z = self.logits(d, u[i] + P); wrong = z.copy(); wrong[:, y[i]] = -np.inf
                obs[i] = u[i] + P[np.argmax(wrong.max(axis=1) - z[:, y[i]])]
        return np.hstack([self.centres[d] + rng.normal(0, 0.03, (n, NI_ID)), obs]), y

# ---------------------------------------------------------------- the specialist library (files + metadata) ----
class Library:
    """library/<id>.model (uai32 model file) + library/<id>.json (metadata).  index.json holds every specialist's
    metadata so routing needs only 21 numbers per specialist in memory (about 1 KB of JSON) and never opens a model file."""
    def __init__(self, root):
        self.root = root; os.makedirs(root, exist_ok=True)
        self.meta = {}; ip = os.path.join(root, 'index.json')
        if os.path.exists(ip): self.meta = json.load(open(ip))
        self._matrix = None
    def path(self, sid): return os.path.join(self.root, sid + '.model')
    def save(self):
        json.dump(self.meta, open(os.path.join(self.root, 'index.json'), 'w'), indent=0)
        for sid, m in self.meta.items(): json.dump(m, open(os.path.join(self.root, sid + '.json'), 'w'), indent=1)
        self._matrix = None
    def register(self, sid, src_model, parent, parent_sha, trained_on, X, y, rng, event):
        """Copy a trained model file into the library and record its metadata, including the familiarity model
        (centroid/spread of the training inputs, 99th-percentile standardised distance) and held-out accuracy."""
        shutil.copyfile(src_model, self.path(sid))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        dist = np.sqrt(np.mean(((X - mu) / sd) ** 2, axis=1))
        k, n = Sentinel.accuracy(self.path(sid), *rng_split(X, y, rng)[1])
        self.meta[sid] = dict(id=sid, version=self.meta.get(sid, {}).get('version', 0) + 1, sha256=sha256(self.path(sid)),
                              bytes=os.path.getsize(self.path(sid)), parent=parent, parent_sha256=parent_sha,
                              trained_on=trained_on, centroid=mu.tolist(), spread=sd.tolist(),
                              radius=float(np.percentile(dist, 99)), val_acc=k / n,
                              history=self.meta.get(sid, {}).get('history', []) + [dict(event=event, sha256=sha256(self.path(sid)))])
        self._matrix = None
    def touch(self, sid, event):                               # after training a specialist's file in place
        self.meta[sid]['sha256'] = sha256(self.path(sid)); self.meta[sid]['version'] += 1
        self.meta[sid]['history'].append(dict(event=event, sha256=self.meta[sid]['sha256'])); self._matrix = None
    def remove(self, sid):
        for ext in ('.model', '.json'):
            p = os.path.join(self.root, sid + ext); os.path.exists(p) and os.remove(p)
        self.meta.pop(sid, None); self._matrix = None
    def verify(self, sid):                                     # provenance: file on disk must match the registered hash
        return os.path.exists(self.path(sid)) and sha256(self.path(sid)) == self.meta[sid]['sha256']
    def matrix(self):
        if self._matrix is None:
            ids = sorted(self.meta); M = self.meta
            self._matrix = (ids, np.array([M[i]['centroid'] for i in ids]), np.array([M[i]['spread'] for i in ids]),
                            np.array([M[i]['radius'] for i in ids]), np.array([M[i]['val_acc'] for i in ids]))
        return self._matrix

def rng_split(X, y, rng, hold=100):
    idx = rng.permutation(len(y)); return (X[idx[hold:]], y[idx[hold:]]), (X[idx[:hold]], y[idx[:hold]])

# ---------------------------------------------------------------- router, arbitration, composition -----------
def route(lib, X, k=3, chunk=256):
    """Familiarity routing.  For every specialist: standardised distance of x to its training centroid, divided by
    its radius (<= 1 means x looks like its training data), and a diagonal-Gaussian score that also charges for the
    specialist's training spread.  Candidates are ranked by that score among the specialists whose region contains
    x.  Cost: N x NI multiply-adds per row; no model file is touched."""
    ids, C, S, R, A = lib.matrix(); logS = np.mean(np.log(S), axis=1)[None]
    orders, rels = [], []
    for i in range(0, len(X), chunk):                                                     # chunked: memory is chunk x N x NI, not rows x N x NI
        z2 = np.mean(((X[i:i + chunk, None, :] - C[None]) / S[None]) ** 2, axis=2)        # chunk x specialists
        rel = np.sqrt(z2) / R[None]                                                       # <= 1: inside the familiar region
        nll = 0.5 * z2 + logS                                                             # diagonal-Gaussian score: a broad training
        order = np.argsort(nll + 1e6 * (rel > 1.0), axis=1)[:, :k]                        # spread is penalised, so "noisy" experts
        orders.append(order); rels.append(np.take_along_axis(rel, order, axis=1))         # do not win by default
    return ids, np.vstack(orders), np.vstack(rels)

def decide(lib, X, k=1, margin=0.75, log=None):
    """Full decision for a batch: route, invoke the top-k candidates inside their radius (k=1: single specialist;
    k>1: arbitration by weighted vote, weight = held-out accuracy x confidence x exp(-relative distance)), expose
    disagreement, mark unsupported inputs.  Specialists are invoked in batches (one process per specialist) and
    every record carries full provenance."""
    t_start = time.perf_counter(); ids, order, rel = route(lib, X, k); t_route = time.perf_counter() - t_start
    sub0 = Sentinel.seconds
    n = len(X); answers = np.full(n, -1); status = ['unsupported'] * n; prov = [None] * n
    jobs = {}                                                  # specialist -> list of row indices to invoke
    for i in range(n):
        for j in range(k):
            if rel[i, j] <= 1.0: jobs.setdefault(order[i, j], []).append(i)
    votes = {}; bad = set()                                    # row -> list of (specialist idx, class, conf, weight)
    for s, rows_ in jobs.items():
        sid = ids[s]
        if not lib.verify(sid): bad.add(s); continue           # corrupt or missing file: this specialist cannot contribute
        cls, conf = Sentinel.predict(lib.path(sid), X[rows_])
        for r, c, p in zip(rows_, cls, conf):
            j = list(order[r]).index(s)
            w = lib.meta[sid]['val_acc'] * p * float(np.exp(-rel[r, j]))
            votes.setdefault(r, []).append((sid, int(c), float(p), w, float(rel[r, j])))
    for r in range(n):
        v = votes.get(r, [])
        rec = dict(selected=ids[order[r, 0]] if rel[r, 0] <= 1 and order[r, 0] not in bad else None,
                   alternatives=[(ids[order[r, j]], round(float(rel[r, j]), 3)) for j in range(1, k)],
                   contributions=[dict(id=s, sha256=lib.meta[s]['sha256'][:16], cls=c, conf=p, weight=round(w, 4), rel_dist=round(d, 3)) for s, c, p, w, d in v])
        if not v:
            rec['status'] = 'unsupported'; rec['nearest_rel_dist'] = round(float(rel[r, 0]), 3)
        else:
            tot = {}
            for _, c, _, w, _ in v: tot[c] = tot.get(c, 0) + w
            best = max(tot, key=tot.get); rest = sorted(tot.values(), reverse=True)
            answers[r] = best; rec['answer'] = best; rec['class_weights'] = tot
            if len(tot) > 1 and rest[1] >= margin * rest[0]:
                rec['status'] = 'disagreement'; rec['disagreement'] = {str(c): round(w, 4) for c, w in tot.items()}
            else: rec['status'] = 'routed'
        status[r] = rec['status']; prov[r] = rec
        if log is not None: log.write(json.dumps(rec) + '\n')
    total = time.perf_counter() - t_start; inv = Sentinel.seconds - sub0
    return answers, status, prov, dict(total=total, routing=t_route, invocation=inv, orchestration=total - t_route - inv)

# ---------------------------------------------------------------- helpers for the stages -------------------
def build_specialist(lib, world, d, canonical, sid, rng, condition='normal', rows=500, epochs=60, lr=0.05, seed=1, event='train'):
    """Clone the canonical brain (byte copy), then train the clone on domain d under one condition."""
    X, y = world.sample(d, rows + 100, rng, condition)
    (Xt, yt), _ = rng_split(X, y, rng)
    tmp = os.path.join(WORK, sid + '.model'); shutil.copyfile(canonical, tmp)
    Sentinel.train(tmp, Xt, yt, epochs, lr, seed)
    lib.register(sid, tmp, parent='canonical', parent_sha=sha256(canonical), event=event, X=X, y=y, rng=rng,
                 trained_on=dict(domain=int(d), condition=condition, rows=rows, epochs=epochs, lr=lr, seed=seed))
    os.remove(tmp)

def make_canonical(path, rng, seed=1):
    """The canonical brain: normalisation statistics from inputs spread over the whole input space, random
    He-uniform weights from SEED, zero epochs.  Every specialist starts as a byte copy of this file."""
    X = rng.uniform(-1, 1, (2000, NI)); y = rng.integers(0, NO, 2000)
    Sentinel.create(path, X, y, NH, seed)

MAXRSS = os.path.join(WORK, 'maxrss')
def spawn_rss_kb(binary, args, stdin_path=os.devnull):
    """Peak RSS of ONE process (KB).  Measured through tools/maxrss.c, a tiny C launcher: the kernel charges a
    vfork/posix_spawn child with its parent's pages at exec time, so spawning from this numpy-sized orchestrator
    reports ~50 MB for everything, even /bin/true.  The C launcher's own footprint is below the uai32 floor."""
    if not os.path.exists(MAXRSS): subprocess.run(['cc', '-O2', os.path.join(HERE, 'tools', 'maxrss.c'), '-o', MAXRSS], check=True)
    out = subprocess.run([MAXRSS, '-i', stdin_path, binary] + args, capture_output=True, text=True, check=True).stdout
    if stdin_path.startswith(WORK + '/rows_') and os.path.exists(stdin_path): os.remove(stdin_path)
    return int(out.split('maxrss')[1].split('KB')[0])

def test_set(world, domains, rng, per=100, condition='normal'):
    Xs, ys, ds = [], [], []
    for d in domains:
        X, y = world.sample(d, per, rng, condition); Xs.append(X); ys.append(y); ds += [d] * per
    return np.vstack(Xs), np.concatenate(ys), np.array(ds)

def metrics(lib, answers, status, prov, y, dom):
    sel_dom = np.array([lib.meta[p['selected']]['trained_on']['domain'] if p['selected'] else -1 for p in prov])
    st = np.array(status)
    return dict(n=int(len(y)), task_acc=float(np.mean(answers == y)), routing_acc=float(np.mean(sel_dom == dom)),
                unsupported=float(np.mean(st == 'unsupported')), disagreement=float(np.mean(st == 'disagreement')),
                acc_when_routed=float(np.mean((answers == y)[st != 'unsupported'])) if (st != 'unsupported').any() else None)

# ================================================================ the stages ==================================
RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
def save(name, obj):
    json.dump(obj, open(os.path.join(RES, name + '.json'), 'w'), indent=1, default=float); print(f'  -> results/{name}.json')

def fresh_library(name):
    root = os.path.join(WORK, name); shutil.rmtree(root, ignore_errors=True); return Library(root)

def stage1(seed):
    """Clone divergence: identical start, different experiences, different states and behaviour, persistence."""
    rng = np.random.default_rng(seed); world = World(2, seed); out = dict(stage=1, seed=seed)
    can = os.path.join(WORK, 'canonical_s1.model'); make_canonical(can, rng, seed)
    A, B, C = [os.path.join(WORK, f'clone_{n}.model') for n in 'ABC']
    for p in (A, B, C): shutil.copyfile(can, p)
    out['initial_sha256'] = dict(canonical=sha256(can), A=sha256(A), B=sha256(B), C=sha256(C))
    out['identical_at_start'] = len(set(out['initial_sha256'].values())) == 1
    XA, yA = world.sample(0, 500, rng); XB, yB = world.sample(1, 500, rng)
    Sentinel.train(A, XA, yA, 60); Sentinel.train(B, XB, yB, 60); Sentinel.train(C, XA, yA, 60)   # C: same experience as A
    out['trained_sha256'] = dict(A=sha256(A), B=sha256(B), C=sha256(C))
    out['A_differs_from_B'] = sha256(A) != sha256(B); out['A_equals_C_same_experience'] = sha256(A) == sha256(C)
    TA, tA = world.sample(0, 300, rng); TB, tB = world.sample(1, 300, rng)
    acc = lambda m, X, y: (lambda k_n: k_n[0] / k_n[1])(Sentinel.accuracy(m, X, y))
    out['accuracy'] = {'A_on_domain0': acc(A, TA, tA), 'A_on_domain1': acc(A, TB, tB), 'B_on_domain0': acc(B, TA, tA), 'B_on_domain1': acc(B, TB, tB),
                       'canonical_on_domain0': acc(can, TA, tA), 'canonical_on_domain1': acc(can, TB, tB)}
    pa, _ = Sentinel.predict(A, TA); pb, _ = Sentinel.predict(B, TA)
    out['fraction_of_domain0_inputs_where_A_and_B_disagree'] = float(np.mean(pa != pb))
    # persistence: every call above was a fresh process reading the file; repeat in new processes and compare bytes
    pa2, _ = Sentinel.predict(A, TA); pb2, _ = Sentinel.predict(B, TB)
    out['restart_identical_outputs'] = bool(np.array_equal(pa, pa2)) and bool(np.array_equal(Sentinel.predict(B, TB)[0], pb2))
    out['sha256_unchanged_by_use'] = sha256(A) == out['trained_sha256']['A'] and sha256(B) == out['trained_sha256']['B']
    out['model_bytes'] = os.path.getsize(A)
    save('stage1', out); return out

def build_library(name, world, domains, rng, seed, canonical, **kw):
    lib = fresh_library(name); t = time.perf_counter()
    for d in domains: build_specialist(lib, world, d, canonical, f's{d:04d}', rng, seed=seed, **kw)
    lib.save(); return lib, time.perf_counter() - t

def stage2(seed):
    """Ten specialists, bounded competences, routing with provenance, unsupported inputs."""
    rng = np.random.default_rng(seed); world = World(13, seed); can = os.path.join(WORK, 'canonical_s2.model'); make_canonical(can, rng, seed)
    lib, build_s = build_library('lib_s2', world, range(10), rng, seed, can)
    X, y, dom = test_set(world, range(10), rng, 100)
    with open(os.path.join(RES, 'stage2_provenance.jsonl'), 'w') as log:
        ans, st, prov, tm = decide(lib, X, k=3, log=log)          # k=3 so alternatives are recorded
    m = metrics(lib, ans, st, prov, y, dom)
    # unsupported inputs: three domains the swarm never saw, and uniform noise
    Xu, yu, du = test_set(world, [10, 11, 12], rng, 100); Xn = rng.uniform(-1, 1, (100, NI))
    au, su, pu, _ = decide(lib, Xu, k=1); an, sn, pn, _ = decide(lib, Xn, k=1)
    out = dict(stage=2, seed=seed, build_seconds=build_s, specialists={s: dict(domain=lib.meta[s]['trained_on']['domain'], val_acc=lib.meta[s]['val_acc'], sha256=lib.meta[s]['sha256'][:16], bytes=lib.meta[s]['bytes'], radius=round(lib.meta[s]['radius'], 3)) for s in sorted(lib.meta)},
               in_domain=m, unknown_domains_unsupported_rate=float(np.mean(np.array(su) == 'unsupported')), noise_unsupported_rate=float(np.mean(np.array(sn) == 'unsupported')),
               example_records=prov[:3] + [p for p in pu if p['status'] == 'unsupported'][:1], per_specialist_accuracy={})
    for s in sorted(lib.meta):
        d = lib.meta[s]['trained_on']['domain']; sel = np.array([p['selected'] == s for p in prov])
        out['per_specialist_accuracy'][s] = dict(rows_routed=int(sel.sum()), accuracy=float(np.mean((ans == y)[sel])) if sel.any() else None, own_domain_rows=int(np.sum(dom == d)))
    save('stage2', out); return out

def stage3(seed):
    """Arbitration between several specialists competent for the same domain; disagreement is exposed."""
    rng = np.random.default_rng(seed); world = World(10, seed); can = os.path.join(WORK, 'canonical_s3.model'); make_canonical(can, rng, seed)
    lib = fresh_library('lib_arb')
    for d in range(3):
        build_specialist(lib, world, d, can, f's{d:04d}a', rng, seed=1, rows=400)
        build_specialist(lib, world, d, can, f's{d:04d}b', rng, seed=2, rows=250)
        build_specialist(lib, world, d, can, f's{d:04d}c', rng, condition='noisy', seed=3, rows=400)
    lib.save(); out = dict(stage=3, seed=seed, experts={s: dict(domain=lib.meta[s]['trained_on']['domain'], condition=lib.meta[s]['trained_on']['condition'], val_acc=lib.meta[s]['val_acc']) for s in sorted(lib.meta)}, results={})
    for cond in ('normal', 'edge'):
        X, y, dom = test_set(world, range(3), rng, 150, cond)
        single = decide(lib, X, k=1); arb = decide(lib, X, k=3, margin=0.75)
        ms, ma = metrics(lib, single[0], single[1], single[2], y, dom), metrics(lib, arb[0], arb[1], arb[2], y, dom)
        st = np.array(arb[1]); dis = st == 'disagreement'
        out['results'][cond] = dict(single_specialist=ms, arbitration_k3=ma, disagreement_rate=float(dis.mean()),
                                    accuracy_on_disagreements=float(np.mean((arb[0] == y)[dis])) if dis.any() else None,
                                    accuracy_on_agreements=float(np.mean((arb[0] == y)[~dis & (st != 'unsupported')])),
                                    example_disagreement=next((p for p in arb[2] if p['status'] == 'disagreement'), None))
        # per-expert accuracy on this condition, for comparison with the arbitrated result
        out['results'][cond]['per_expert'] = {s: float(np.mean(Sentinel.predict(lib.path(s), X[dom == lib.meta[s]['trained_on']['domain']])[0] == y[dom == lib.meta[s]['trained_on']['domain']])) for s in sorted(lib.meta)}
    save('stage3', out); return out

def stage4(seed):
    """Composition: a compound reading from two devices; the answer is a table lookup of both device states."""
    rng = np.random.default_rng(seed); world = World(10, seed); can = os.path.join(WORK, 'canonical_s4.model'); make_canonical(can, rng, seed)
    lib, _ = build_library('lib_comp', world, range(10), rng, seed, can)
    table = rng.integers(0, NO, (NO, NO))
    while len(set(table.flatten())) < NO: table = rng.integers(0, NO, (NO, NO))
    def compound(n):
        da, db = rng.integers(0, 10, n), rng.integers(0, 10, n); Xa, ya, Xb, yb = [], [], [], []
        for a, b in zip(da, db):
            x1, y1 = world.sample(a, 1, rng); x2, y2 = world.sample(b, 1, rng); Xa.append(x1[0]); ya.append(y1[0]); Xb.append(x2[0]); yb.append(y2[0])
        Xa, Xb, ya, yb = np.array(Xa), np.array(Xb), np.array(ya), np.array(yb)
        return np.hstack([Xa, Xb]), table[ya, yb], (Xa, Xb, ya, yb)
    Xtr, ytr, _ = compound(5000); Xte, yte, (Xa, Xb, ya, yb) = compound(500)
    # the swarm: route each half, invoke, compose by the table (the composition layer knows the table, not the device rules)
    t = time.perf_counter(); a_ans, a_st, a_prov, _ = decide(lib, Xa, k=1); b_ans, b_st, b_prov, _ = decide(lib, Xb, k=1)
    ok = (np.array(a_st) != 'unsupported') & (np.array(b_st) != 'unsupported')
    comp = np.where(ok, table[np.maximum(a_ans, 0), np.maximum(b_ans, 0)], -1); swarm_s = time.perf_counter() - t
    out = dict(stage=4, seed=seed, table=table.tolist(), swarm=dict(accuracy=float(np.mean(comp == yte)), unsupported=float(np.mean(~ok)), stage_a_accuracy=float(np.mean(a_ans == ya)), stage_b_accuracy=float(np.mean(b_ans == yb)), seconds=swarm_s,
               example=dict(input_halves=[a_prov[0]['selected'], b_prov[0]['selected']], device_states=[int(a_ans[0]), int(b_ans[0])], composed=int(comp[0]), truth=int(yte[0]))), baselines={})
    for nh in (16, 64, 256):                                    # a single brain trained directly on the compound task
        m = os.path.join(WORK, f'compound_{nh}.model'); Sentinel.create(m, Xtr, ytr, nh, seed); Sentinel.train(m, Xtr, ytr, 100)
        k, n = Sentinel.accuracy(m, Xte, yte); out['baselines'][f'single_brain_20-{nh}-4'] = dict(accuracy=k / n, bytes=os.path.getsize(m), train_rows=len(ytr), epochs=100)
    # same supervision as the swarm but ONE brain: a 10-16-4 brain trained on every domain's device-state labels (the
    # ten specialists' training rows pooled), applied to each half, composed through the same table
    Xp, yp = [], []
    for d in range(10): Xd, yd = world.sample(d, 500, rng); Xp.append(Xd); yp.append(yd)
    Xp, yp = np.vstack(Xp), np.concatenate(yp); m = os.path.join(WORK, 'compound_onebrain.model'); Sentinel.create(m, Xp, yp, NH, seed); Sentinel.train(m, Xp, yp, 60)
    pa, _ = Sentinel.predict(m, Xa); pb, _ = Sentinel.predict(m, Xb)
    out['baselines']['one_brain_10-16-4_on_device_states_then_table'] = dict(accuracy=float(np.mean(table[pa, pb] == yte)), stage_a_accuracy=float(np.mean(pa == ya)), stage_b_accuracy=float(np.mean(pb == yb)), bytes=os.path.getsize(m), train_rows=int(len(yp)), epochs=60)
    out['disclosure'] = 'the composition layer is given the 4x4 table (an oracle); the specialists were trained with device-state labels (intermediate supervision); the direct baselines only ever saw composed labels'
    out['single_specialist_alone'] = 'structurally impossible: a specialist takes 10 inputs, the compound input has 20'
    save('stage4', out); return out

def measure_library(lib, X, label):
    """Stage 5/6 resource measurements for an existing library."""
    ids = sorted(lib.meta); root = lib.root
    storage = sum(os.path.getsize(os.path.join(root, f)) for f in os.listdir(root))
    t = time.perf_counter(); route(lib, X[:1000]); route_s = (time.perf_counter() - t) / min(1000, len(X))
    times = []
    for i in range(30):                                        # single-query path: fresh process, load one file, one row
        t = time.perf_counter(); Sentinel.predict(lib.path(ids[i % len(ids)]), X[i:i + 1]); times.append(time.perf_counter() - t)
    return dict(label=label, specialists=len(ids), per_specialist_model_bytes=lib.meta[ids[0]]['bytes'], per_specialist_meta_bytes=os.path.getsize(os.path.join(root, ids[0] + '.json')),
                index_bytes=os.path.getsize(os.path.join(root, 'index.json')), storage_bytes=storage, routing_seconds_per_row=route_s,
                load_and_predict_seconds_median=float(np.median(times)), load_and_predict_seconds_min=float(np.min(times)),
                specialist_process_peak_rss_kb=spawn_rss_kb(UAI32, ['predict', lib.path(ids[0])], tmpfile(rows_text(X[:1]))), true_process_peak_rss_kb=spawn_rss_kb('/bin/true', []),
                specialist_working_set_bytes=4 * (2 * NI + NH * (NI + 1) + NO * (NH + 1)) + 4 * (NI + 2 * NH + 2 * NO) + 4 * (NI + 1),   # float32 parameters + activations + one row, from uai32.c
                orchestrator_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resident_specialists_between_calls_by_construction=0)

def stage5(seed):
    rng = np.random.default_rng(seed); world = World(10, seed); can = os.path.join(WORK, 'canonical_s5.model'); make_canonical(can, rng, seed)
    lib, _ = build_library('lib_s5', world, range(10), rng, seed, can); X, y, dom = test_set(world, range(10), rng, 100)
    out = dict(stage=5, seed=seed, measurements=measure_library(lib, X, '10 specialists'))
    ans, st, prov, tm = decide(lib, X, k=1); out['batch'] = dict(rows=len(X), active_specialists_during_batch=len(set(p['selected'] for p in prov if p['selected'])), seconds=tm, processes=Sentinel.calls)
    save('stage5', out); return out

def stage6(seed, sizes=(2, 10, 100, 1000), big_cap=2048):
    out = dict(stage=6, seed=seed, levels=[])
    for N in sizes:
        rng = np.random.default_rng(seed + N); world = World(N, seed + N); can = os.path.join(WORK, f'canonical_s6_{N}.model'); make_canonical(can, rng, seed)
        Sentinel.calls = 0; Sentinel.seconds = 0.0; Sentinel.failures = 0
        lib, build_s = build_library(f'lib_s6_{N}', world, range(N), rng, seed, can)
        X, y, dom = test_set(world, range(N), rng, 50); calls0 = Sentinel.calls
        ans, st, prov, tm = decide(lib, X, k=1)
        lvl = dict(N=N, build_seconds=build_s, train_processes=N, **metrics(lib, ans, st, prov, y, dom), eval_rows=len(X), eval_seconds=tm['total'], routing_seconds=tm['routing'],
                   invocation_seconds=tm['invocation'], orchestration_seconds=tm['orchestration'], eval_processes=Sentinel.calls - calls0, failed_invocations=Sentinel.failures, resources=measure_library(lib, X, f'{N} specialists'))
        Xo = rng.uniform(-1, 1, (300, NI)); _, so, _, _ = decide(lib, Xo, k=1); lvl['out_of_distribution_unsupported_rate'] = float(np.mean(np.array(so) == 'unsupported'))
        # monolith baselines on the union of the specialists' training data: the SAME rows per domain (500) and the
        # same epochs (60) as each specialist, except the capped equal-storage monolith at N >= 1000 (20 epochs,
        # recorded).  "Equal storage" = N x the 576-byte model file (metadata excluded; file = 96 + 30*NH bytes).
        Xall, yall = [], []
        for d in range(N):
            Xd, yd = world.sample(d, 500, rng); Xall.append(Xd); yall.append(yd)
        Xall, yall = np.vstack(Xall), np.concatenate(yall)
        nh_eq = min(big_cap, max(NH, (N * lib.meta[sorted(lib.meta)[0]]['bytes'] - 96) // 30))
        for name, nh, epochs in (('mini_monolith_10-16-4', NH, 60), ('equal_storage_monolith', nh_eq, 60 if nh_eq < big_cap else 20)):
            m = os.path.join(WORK, f'mono_{N}_{nh}.model'); t = time.perf_counter(); Sentinel.create(m, Xall, yall, nh, seed); Sentinel.train(m, Xall, yall, epochs)
            k, n = Sentinel.accuracy(m, X, y)
            lvl[name] = dict(hidden=int(nh), bytes=os.path.getsize(m), accuracy=k / n, train_rows=int(len(yall)), rows_per_domain=500, epochs=epochs, lr=0.05, capped=bool(nh_eq >= big_cap and name != 'mini_monolith_10-16-4'),
                             train_seconds=time.perf_counter() - t, macs_per_query=int(nh * (NI + 1) + NO * (nh + 1)))
        # swarm MACs: routing = 2 per feature per specialist (subtract and scale, square-accumulate) + one specialist forward pass
        lvl['swarm_macs_per_query'] = int(N * 2 * NI + NH * (NI + 1) + NO * (NH + 1)); lvl['swarm_storage_bytes'] = lvl['resources']['storage_bytes']
        lvl['swarm_model_bytes_total'] = int(N * lib.meta[sorted(lib.meta)[0]]['bytes'])
        out['levels'].append(lvl); save('stage6', out)
        print(f'  N={N}: task acc {lvl["task_acc"]:.3f} routing {lvl["routing_acc"]:.3f} unsupported {lvl["unsupported"]:.3f} | mini monolith {lvl["mini_monolith_10-16-4"]["accuracy"]:.3f} | equal-storage {lvl["equal_storage_monolith"]["accuracy"]:.3f} (NH={lvl["equal_storage_monolith"]["hidden"]})')
    return out

def stage7(seed):
    """Populations: five experts for ONE domain trained under different conditions; can the swarm pick the right one?"""
    rng = np.random.default_rng(seed); world = World(10, seed); can = os.path.join(WORK, 'canonical_s7.model'); make_canonical(can, rng, seed)
    lib = fresh_library('lib_pop')
    for c in CONDITIONS: build_specialist(lib, world, 0, can, f'expert_{c}', rng, condition=c, rows=400, seed=1)
    lib.save(); out = dict(stage=7, seed=seed, matrix={}, swarm={})
    for c in CONDITIONS:
        X, y, dom = test_set(world, [0], rng, 300, c)
        out['matrix'][c] = {e: float(np.mean(Sentinel.predict(lib.path(e), X)[0] == y)) for e in sorted(lib.meta)}
        a1, s1, p1, _ = decide(lib, X, k=1); a3, s3, p3, _ = decide(lib, X, k=3)
        sel = {}
        for p in p1: sel[p['selected']] = sel.get(p['selected'], 0) + 1
        out['swarm'][c] = dict(selected_histogram=sel, accuracy_k1=float(np.mean(a1 == y)), accuracy_k3=float(np.mean(a3 == y)), unsupported=float(np.mean(np.array(s1) == 'unsupported')),
                               oracle_best_expert=max(out['matrix'][c], key=out['matrix'][c].get), oracle_best_accuracy=max(out['matrix'][c].values()), normal_expert_accuracy=out['matrix'][c]['expert_normal'])
    save('stage7', out); return out

def stage8(seed):
    """Learning over time: isolation, forgetting, retraining, replacing, cloning, branching."""
    rng = np.random.default_rng(seed); world = World(10, seed); can = os.path.join(WORK, 'canonical_s8.model'); make_canonical(can, rng, seed)
    lib, _ = build_library('lib_learn', world, range(10), rng, seed, can); out = dict(stage=8, seed=seed)
    shas = {s: lib.meta[s]['sha256'] for s in lib.meta}; T3 = world.sample(3, 300, rng); T7 = world.sample(7, 300, rng)
    acc = lambda s, T: (lambda kn: kn[0] / kn[1])(Sentinel.accuracy(lib.path(s), *T))
    # (a) continued training of one specialist on new data of its own domain; nothing else changes
    before = acc('s0003', T3); Xn, yn = world.sample(3, 200, rng); Sentinel.train(lib.path('s0003'), Xn, yn, 20); lib.touch('s0003', 'continue')
    out['continued_training'] = dict(s0003_before=before, s0003_after=acc('s0003', T3), s0003_sha_changed=lib.meta['s0003']['sha256'] != shas['s0003'],
                                     others_unchanged=all(sha256(lib.path(s)) == shas[s] for s in lib.meta if s != 's0003'))
    # (b) catastrophic forgetting inside one specialist: train s0003 on domain 7's task
    X7, y7 = world.sample(7, 400, rng); Sentinel.train(lib.path('s0003'), X7, y7, 40); lib.touch('s0003', 'continue-other-domain')
    out['forgetting'] = dict(s0003_on_domain3_after_learning_domain7=acc('s0003', T3), s0003_on_domain7=acc('s0003', T7), note='no replay or regularisation exists in uai32; this is the raw effect')
    # (c) retrain from the canonical brain
    build_specialist(lib, world, 3, can, 's0003', rng, seed=seed, event='retrain'); out['retrained'] = dict(s0003_on_domain3=acc('s0003', T3), version=lib.meta['s0003']['version'])
    # (d) a failed expert: corrupt s0005 on disk; the router must notice and the swarm must not answer from it
    b = bytearray(open(lib.path('s0005'), 'rb').read()); b[40] ^= 0xFF; open(lib.path('s0005'), 'wb').write(b)
    X5, y5, d5 = test_set(world, [5], rng, 100); a, st, pv, _ = decide(lib, X5, k=1)
    out['failed_expert'] = dict(verify_false=not lib.verify('s0005'), rows_unsupported_while_corrupt=float(np.mean(np.array(st) == 'unsupported')))
    build_specialist(lib, world, 5, can, 's0005', rng, seed=seed, event='replace'); a, st, pv, _ = decide(lib, X5, k=1)
    out['failed_expert'].update(accuracy_after_replacement=float(np.mean(a == y5)), version=lib.meta['s0005']['version'])
    # (e) clone a successful expert, (f) branch it into two differently trained descendants
    shutil.copyfile(lib.path('s0002'), os.path.join(WORK, 'clone2.model'))
    lib.meta['s0002_clone'] = dict(lib.meta['s0002'], id='s0002_clone', parent='s0002', parent_sha256=lib.meta['s0002']['sha256'], history=[dict(event='clone', sha256=lib.meta['s0002']['sha256'])])
    shutil.copyfile(lib.path('s0002'), lib.path('s0002_clone')); lib.save()
    T2 = world.sample(2, 300, rng); out['clone'] = dict(same_sha=sha256(lib.path('s0002_clone')) == lib.meta['s0002']['sha256'], identical_outputs=bool(np.array_equal(Sentinel.predict(lib.path('s0002'), T2[0])[0], Sentinel.predict(lib.path('s0002_clone'), T2[0])[0])))
    branches = {}
    for name, cond in (('s0002_noisy', 'noisy'), ('s0002_shifted', 'shifted')):
        shutil.copyfile(lib.path('s0002'), os.path.join(WORK, name + '.model')); Xc, yc = world.sample(2, 500, rng, cond); (Xt, yt), _ = rng_split(Xc, yc, rng)
        Sentinel.train(os.path.join(WORK, name + '.model'), Xt, yt, 40, seed=5)
        lib.register(name, os.path.join(WORK, name + '.model'), parent='s0002', parent_sha=lib.meta['s0002']['sha256'], event='branch', X=Xc, y=yc, rng=rng, trained_on=dict(domain=2, condition=cond, rows=400, epochs=40, lr=0.05, seed=5))
        branches[name] = lib.meta[name]['sha256'][:16]
    lib.save(); Tn = world.sample(2, 300, rng, 'noisy'); Ts = world.sample(2, 300, rng, 'shifted')
    out['branch'] = dict(parent_sha=lib.meta['s0002']['sha256'][:16], branches=branches, all_distinct=len({lib.meta['s0002']['sha256'], *[lib.meta[b]['sha256'] for b in branches]}) == 3,
                         accuracy=dict(parent_on_noisy=acc('s0002', Tn), noisy_branch_on_noisy=acc('s0002_noisy', Tn), parent_on_shifted=acc('s0002', Ts), shifted_branch_on_shifted=acc('s0002_shifted', Ts), noisy_branch_on_shifted=acc('s0002_noisy', Ts)))
    out['removal'] = dict(before=len(lib.meta)); lib.remove('s0002_clone'); lib.save(); out['removal']['after'] = len(lib.meta); out['removal']['file_gone'] = not os.path.exists(lib.path('s0002_clone'))
    save('stage8', out); return out

def stretch(seed):
    """Expert spawning: inputs nobody handles are buffered with their (externally supplied) labels; once enough exist,
    a new specialist is commissioned from the canonical brain."""
    rng = np.random.default_rng(seed); world = World(12, seed); can = os.path.join(WORK, 'canonical_stretch.model'); make_canonical(can, rng, seed)
    lib, _ = build_library('lib_spawn', world, range(10), rng, seed, can); out = dict(stage='stretch', seed=seed, supervision='labels for buffered inputs come from the environment (the synthetic teacher), i.e. external supervision, disclosed')
    buffer_X, buffer_y, spawned = [], [], []
    for step in range(5):
        Xq, yq, _ = test_set(world, [10, 11], rng, 60)          # two unseen domains keep arriving
        ans, st, prov, _ = decide(lib, Xq, k=1); uns = np.array(st) == 'unsupported'
        buffer_X += list(Xq[uns]); buffer_y += list(yq[uns])
        rec = dict(step=step, unsupported_rate=float(uns.mean()), accuracy_on_supported=float(np.mean((ans == yq)[~uns])) if (~uns).any() else None, buffer=len(buffer_y))
        if len(buffer_y) >= 200:                                 # commission a specialist for the buffered niche (cluster by identifying features)
            B = np.array(buffer_X); by = np.array(buffer_y); centre = B[:, :NI_ID].mean(0); near = np.linalg.norm(B[:, :NI_ID] - centre, axis=1) < 0.1
            if near.sum() < 100: near = np.linalg.norm(B[:, :NI_ID] - B[0, :NI_ID], axis=1) < 0.1
            sid = f'spawned_{len(spawned)}'; tmp = os.path.join(WORK, sid + '.model'); shutil.copyfile(can, tmp)
            (Xt, yt), _ = rng_split(B[near], by[near], rng, hold=min(50, int(near.sum() // 4))); Sentinel.train(tmp, Xt, yt, 40)
            lib.register(sid, tmp, 'canonical', sha256(can), dict(domain='spawned', condition='buffered', rows=int(near.sum()), epochs=40, lr=0.05, seed=1), B[near], by[near], rng, 'spawn'); lib.save()
            spawned.append(sid); rec['spawned'] = sid; rec['from_rows'] = int(near.sum())
            keep = ~near; buffer_X, buffer_y = list(B[keep]), list(by[keep])
        out.setdefault('steps', []).append(rec)
    Xf, yf, df = test_set(world, [10, 11], rng, 100); ans, st, prov, _ = decide(lib, Xf, k=1)
    out['final'] = dict(spawned=spawned, unsupported_rate=float(np.mean(np.array(st) == 'unsupported')), accuracy=float(np.mean(ans == yf)), accuracy_domain10=float(np.mean((ans == yf)[df == 10])), accuracy_domain11=float(np.mean((ans == yf)[df == 11])))
    save('stretch', out); return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stages', nargs='+'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--sizes', default='2,10,100,1000'); a = ap.parse_args()
    if not os.path.exists(UAI32): sys.exit(f'uai32 binary not found at {UAI32}')
    print(f'uai32: {UAI32} ({os.path.getsize(UAI32)} bytes, sha256 {sha256(UAI32)[:16]})')
    for s in a.stages:
        t = time.perf_counter(); print(f'== stage {s}')
        {'1': stage1, '2': stage2, '3': stage3, '4': stage4, '5': stage5, '7': stage7, '8': stage8, 'stretch': stretch,
         '6': lambda seed: stage6(seed, tuple(int(x) for x in a.sizes.split(',')))}[s](a.seed)
        print(f'   {time.perf_counter() - t:.1f} s, {Sentinel.calls} uai32 processes so far')
