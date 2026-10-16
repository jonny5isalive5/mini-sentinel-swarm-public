/* uai32.c -- Sentovara uAI-32: the smallest genuinely trainable neural network we could make.
 *
 * A one-hidden-layer multilayer perceptron (MLP) classifier, trained by plain stochastic
 * gradient descent with backpropagation (softmax output, cross-entropy loss).  Everything
 * the program learns lives in ONE array of floats (P) which is saved to and loaded from a
 * small file.  There are no rules, lookup tables or dataset-specific code anywhere.
 *
 *   uai32 train DATA MODEL [HIDDEN] [EPOCHS] [RATE] [SEED]   learn from DATA, save to MODEL
 *   uai32 test  DATA MODEL                                   accuracy + loss of MODEL on DATA
 *   uai32 predict MODEL  < rows                              print class + probabilities per row
 *
 * If MODEL already exists, `train` loads it and keeps training it (HIDDEN is then ignored,
 * pass anything; SEED still seeds the shuffle).  Delete the file to start from scratch.
 *
 * DATA format: one example per line, whitespace-separated numbers: the features, then the
 * integer class label (0,1,2,...).  The number of features is taken from the first line.
 * Blank lines and lines that do not start with a number are skipped.  `predict` reads rows
 * from stdin and ignores extra numeric columns after the features, so a labelled file can be piped in.
 *
 * MODEL format (all little-endian):  u16 magic 0xA132, u16 NI, u16 NH, u16 NO, then the
 * parameters in the order mean[NI], scale[NI], W1[NH][NI+1], W2[NO][NH+1] (last column = bias).
 * mean and scale are stored as exact float32; every weight as bfloat16 (the top 16 bits of
 * an IEEE-754 float, rounded to nearest even), which halves the file at no accuracy cost.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <float.h>
#include <limits.h>

static int NI, NH, NO;                 /* inputs, hidden units, outputs (classes)       */
static float *P, *mean, *scale, *W1, *W2;  /* all learned parameters live in P           */
static float *xn, *hid, *z, *out, *dh, lse;  /* normalised input, hidden, logits, softmax, hidden grad, log-sum-exp */
static unsigned rng = 1;               /* xorshift32 state; SEED sets it                */

static void die(const char *m) { fprintf(stderr, "uai32: %s\n", m); exit(1); }

static unsigned integer(const char *s, unsigned max) {
    char *e; unsigned long v; errno = 0; v = strtoul(s, &e, 10);
    if (*s < '0' || *s > '9' || *e || errno || v > max) die("invalid integer argument");
    return (unsigned)v;
}

static float rate(const char *s) {
    char *e; float v; errno = 0; v = strtof(s, &e);
    if (e == s || *e || errno || !isfinite(v) || v <= 0 || v > 1e6f) die("invalid RATE");
    return v;
}

static float frand(void) {             /* uniform in [-1, 1) */
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (int)rng * (1.0f / 2147483648.0f);
}

static int nparams(void) { return 2 * NI + NH * (NI + 1) + NO * (NH + 1); }

static void valid_params(void) {
    int i;
    for (i = 0; i < nparams(); i++)
        if (!isfinite(P[i]) || (i >= NI && i < 2 * NI && P[i] < 0)) die("invalid model parameter");
}

static void alloc(void) {
    if (NI < 1 || NH < 1 || NO < 1 || (double)NH * (NI + 1) + (double)NO * (NH + 1) + 2.0 * NI > 1e8) die("bad size");
    P = calloc(nparams(), sizeof *P);
    xn = calloc(NI + 2 * NH + 2 * NO, sizeof *xn);
    if (!P || !xn) die("out of memory");
    mean = P; scale = mean + NI; W1 = scale + NI; W2 = W1 + NH * (NI + 1);
    hid = xn + NI; dh = hid + NH; z = dh + NH; out = z + NO;
}

/* Forward pass: fills xn, hid (ReLU), z (logits), lse, out (softmax probabilities); returns the argmax class. */
static int forward(const float *x) {
    int i, j, k, best = 0; float m = -1e30f, s = 0;
    for (i = 0; i < NI; i++) { xn[i] = (x[i] - mean[i]) * scale[i]; if (!isfinite(xn[i])) die("numeric overflow"); }
    for (j = 0; j < NH; j++) {
        float *w = W1 + j * (NI + 1), a = w[NI];
        for (i = 0; i < NI; i++) a += w[i] * xn[i];
        if (!isfinite(a)) die("numeric overflow");
        hid[j] = a > 0 ? a : 0;
    }
    for (k = 0; k < NO; k++) {
        float *w = W2 + k * (NH + 1), a = w[NH];
        for (j = 0; j < NH; j++) a += w[j] * hid[j];
        if (!isfinite(a)) die("numeric overflow");
        z[k] = a; if (a > m) { m = a; best = k; }
    }
    for (k = 0; k < NO; k++) s += expf(z[k] - m);
    lse = m + logf(s);                                         /* log of the softmax denominator, computed stably */
    if (!isfinite(lse)) die("numeric overflow");
    for (k = 0; k < NO; k++) out[k] = expf(z[k] - lse);
    return best;
}

/* One SGD step on example (x, y): backpropagate the cross-entropy gradient, update P in place. */
static void learn(const float *x, int y, float lr) {
    int i, j, k;
    forward(x);
    for (j = 0; j < NH; j++) dh[j] = 0;
    for (k = 0; k < NO; k++) {
        float g = out[k] - (k == y), *w = W2 + k * (NH + 1);   /* dLoss/dlogit_k */
        for (j = 0; j < NH; j++) { dh[j] += g * w[j]; w[j] -= lr * g * hid[j]; }
        w[NH] -= lr * g;
    }
    for (j = 0; j < NH; j++) if (hid[j] > 0) {                 /* ReLU gate */
        float g = dh[j], *w = W1 + j * (NI + 1);
        for (i = 0; i < NI; i++) w[i] -= lr * g * xn[i];
        w[NI] -= lr * g;
    }
}

/* ---- model file ---- */
static void put16(unsigned v, FILE *f) { putc(v & 255, f); putc(v >> 8 & 255, f); }
static int  get16(FILE *f) { int lo = getc(f), hi = getc(f); if (hi < 0) die("truncated model file"); return lo | hi << 8; }
static unsigned bits(float v) { unsigned b; memcpy(&b, &v, 4); return b; }
static float   value(unsigned b) { float v; memcpy(&v, &b, 4); return v; }

static void save(const char *path) {
    valid_params();
    for (int i = 2 * NI; i < nparams(); i++)
        if (!isfinite(value((bits(P[i]) + 0x7FFF + (bits(P[i]) >> 16 & 1)) & 0xFFFF0000u))) die("model rounding overflow");
    FILE *f = fopen(path, "wb"); int i, n = nparams();
    if (!f) die("cannot write model file");
    put16(0xA132, f); put16(NI, f); put16(NH, f); put16(NO, f);
    for (i = 0; i < 2 * NI; i++) { put16(bits(P[i]) & 0xFFFF, f); put16(bits(P[i]) >> 16, f); }   /* mean, scale: float32 */
    for (; i < n; i++) put16((bits(P[i]) + 0x7FFF + (bits(P[i]) >> 16 & 1)) >> 16, f);         /* weights: bfloat16, round to nearest even */
    if (fclose(f)) die("cannot write model file");
}

static int load(const char *path) {                           /* returns 0 if the file does not exist */
    FILE *f = fopen(path, "rb"); int i, n;
    if (!f) { if (errno == ENOENT) return 0; die("cannot open model file"); }
    if (get16(f) != 0xA132) die("not a uai32 model file");
    NI = get16(f); NH = get16(f); NO = get16(f); alloc();
    if (fseek(f, 0, SEEK_END) || ftell(f) != 8L + 8L * NI + 2L * (nparams() - 2 * NI) || fseek(f, 8, SEEK_SET)) die("wrong model size");
    for (i = 0, n = nparams(); i < 2 * NI; i++) { unsigned lo = get16(f); P[i] = value(lo | (unsigned)get16(f) << 16); }
    for (; i < n; i++) P[i] = value((unsigned)get16(f) << 16);
    fclose(f); valid_params(); return 1;
}

/* ---- data ---- */
static int parse(char *s, float *v, int max, double *label) { /* numbers in s -> v; label kept as double before integer validation */
    int n = 0; char *e;
    for (;;) {
        while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
        if (!*s) break;
        errno = 0; double x = strtod(s, &e);
        if (e == s) { if (n) die("invalid data token"); break; }
        if (errno || !isfinite(x) || x > FLT_MAX || x < -FLT_MAX || (*e && *e != ' ' && *e != '\t' && *e != '\r' && *e != '\n')) die("invalid data number");
        if (n < max) v[n] = (float)x;
        if (label && n == max - 1) *label = x;
        if (n == INT_MAX) die("too many columns");
        n++; s = e;
    }
    return n;
}

static float *X; static int *Y, N;                            /* the loaded dataset */
static void read_data(const char *path, int need_labels) {    /* fixes NI if it is still 0 */
    FILE *f = path ? fopen(path, "r") : stdin; char *line = 0; size_t cap = 0; float *row = 0; int room = 0, n;
    if (!f) die("cannot open data file");
    while (getline(&line, &cap, f) > 0) {
        if (!(n = parse(line, 0, 0, 0))) continue;           /* skip blank / non-numeric lines */
        if (!NI) { if (n < 2) die("need at least one feature and a label"); NI = n - 1; }
        if (need_labels ? n != NI + 1 : n < NI) die("wrong number of columns");   /* predict ignores extra columns */
        if (!row && !(row = calloc(NI + 1, sizeof *row))) die("out of memory");
        if (N == room) {
            if (room > INT_MAX / 2) die("too many rows");
            room = room ? 2 * room : 256;
            X = realloc(X, (size_t)room * NI * sizeof *X); Y = realloc(Y, room * sizeof *Y);
            if (!X || !Y) die("out of memory");
        }
        double label = 0; parse(line, row, NI + 1, &label);
        for (n = 0; n < NI; n++) if (!(row[n] - row[n] == 0)) die("feature is not a finite number");   /* NaN or inf */
        memcpy(X + (size_t)N * NI, row, NI * sizeof *X);
        if (need_labels) {
            if (label < 0 || label > 65534 || label != (int)label) die("labels must be integers in 0..65534");
            Y[N] = (int)label;
        }
        N++;
    }
    if (path) fclose(f);                                      /* memory is reclaimed at exit; no free() needed */
    if (!N) die("no examples");
}

/* Accuracy and mean cross-entropy of the current model over the loaded data (no learning). */
static void report(const char *tag) {
    int i, ok = 0; double loss = 0;
    for (i = 0; i < N; i++) { ok += forward(X + (size_t)i * NI) == Y[i]; loss += lse - z[Y[i]]; }   /* -log p_y */
    if (!isfinite(loss)) die("numeric overflow");
    printf("%s loss %.4f  accuracy %d/%d = %.1f%%\n", tag, loss / N, ok, N, 100.0 * ok / N);
}

static void usage(void) {
    die("usage:\n  uai32 train DATA MODEL [HIDDEN=16] [EPOCHS=50] [RATE=0.05] [SEED=1 (integer)]\n"
        "  uai32 test DATA MODEL\n  uai32 predict MODEL < rows");
}

int main(int argc, char **argv) {
    int i, j, e;
    if (argc < 3) usage();
    if (!strcmp(argv[1], "predict")) {                         /* ---- inference from stdin ---- */
        if (!load(argv[2])) die("cannot open model file");
        read_data(0, 0);
        for (i = 0; i < N; i++) {
            printf("%d", forward(X + (size_t)i * NI));
            for (j = 0; j < NO; j++) printf(" %.3f", out[j]);
            printf("\n");
        }
        return 0;
    }
    if (argc < 4) usage();
    if (!strcmp(argv[1], "test")) {                            /* ---- evaluate a saved model ---- */
        if (!load(argv[3])) die("cannot open model file");
        read_data(argv[2], 1);
        for (i = 0; i < N; i++) if (Y[i] >= NO) die("label outside the model's classes");
        report("test");
        return 0;
    }
    if (strcmp(argv[1], "train")) usage();                     /* ---- train (fresh or continued) ---- */
    int hidden = argc > 4 ? integer(argv[4], 65535) : 16, epochs = argc > 5 ? integer(argv[5], INT_MAX - 1u) : 50;
    float lr = argc > 6 ? rate(argv[6]) : 0.05f;
    rng = argc > 7 ? integer(argv[7], 4294967295u) : 1; if (!rng) rng = 1;
    int resumed = load(argv[3]);
    read_data(argv[2], 1);
    if (!resumed) {
        NH = hidden; NO = 0;
        for (i = 0; i < N; i++) if (Y[i] >= NO) NO = Y[i] + 1;
        if (NI > 65535 || NH > 65535 || NO > 65535) die("bad size");
        alloc();
        for (j = 0; j < NI; j++) {                             /* feature normalisation: shift by the mean, scale by the range */
            float lo = X[j], hi = X[j]; double s = 0;
            for (i = 0; i < N; i++) { float v = X[(size_t)i * NI + j]; s += v; if (v < lo) lo = v; if (v > hi) hi = v; }
            float range = hi - lo;                             /* finite inputs can still overflow here (3e38 - -3e38) */
            if (!(range <= FLT_MAX)) die("feature range overflows float");
            mean[j] = s / N; scale[j] = range > 0 ? 1 / range : 0;
        }
        for (i = 0; i < NH * (NI + 1); i++) W1[i] = (i + 1) % (NI + 1) ? frand() * sqrtf(6.0f / NI) : 0;  /* He-uniform, zero bias */
        for (i = 0; i < NO * (NH + 1); i++) W2[i] = (i + 1) % (NH + 1) ? frand() * sqrtf(6.0f / NH) : 0;
        printf("new model: %d inputs, %d hidden, %d classes, %d parameters\n", NI, NH, NO, nparams());
    } else {
        for (i = 0; i < N; i++) if (Y[i] >= NO) die("label outside the model's classes");
        printf("continuing: %d inputs, %d hidden, %d classes\n", NI, NH, NO);
    }
    int *idx = calloc(N, sizeof *idx); if (!idx) die("out of memory");
    for (i = 0; i < N; i++) idx[i] = i;
    for (e = 1; e <= epochs; e++) {
        for (i = N - 1; i > 0; i--) { frand(); j = rng % (i + 1); int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }  /* shuffle */
        for (i = 0; i < N; i++) learn(X + (size_t)idx[i] * NI, Y[idx[i]], lr);
        printf("epoch %d/%d  ", e, epochs); report("train");
    }
    save(argv[3]);
    return 0;
}
