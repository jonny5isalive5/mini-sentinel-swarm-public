# µAI-32 v1.0.0 (the Mini Sentinel brain), vendored

`dist/uai32` is the exact executable released as **uai32-v1.0.0** in `the private origin repository`
(PRIVATE_ORIGIN, release commit `e651c1c`):
9,188 bytes, SHA-256 `996d73cba9676e8932206b9c80493becde6d04b6a57b70839999b0824c0186cb` (see `dist/SHA256SUMS`).
It is the only thing that learns or predicts anywhere in this project, and it is used unmodified.

`uai32.c` is the matching source (229 lines, C99). The release was built on x86-64 Linux with
gcc 13.3 / binutils 2.42 / glibc 2.39 using the flags in the original `uai32/Makefile`:

```
cc -std=c99 -Os -Wall -Wextra -pedantic -ffp-contract=off -U_FORTIFY_SOURCE -fno-asynchronous-unwind-tables \
   -fno-stack-protector -fno-ident -fno-pie -fno-plt -fcf-protection=none -fno-math-errno \
   -ffunction-sections -fdata-sections uai32.c -o uai32 -s -no-pie -Wl,--gc-sections -Wl,-z,now \
   -Wl,--build-id=none -Wl,-z,norelro -Wl,-z,noseparate-code -Wl,--hash-style=gnu -Wl,-z,max-page-size=4096 \
   -Wl,--no-eh-frame-hdr -Wl,-z,nosectionheader -lm
```

Its full evidence and data are outside this minimal vendored dependency; its original release is retained privately. The µAI-32 publication copy provides separately redacted audit evidence; this copy exists so that `flat-swarm-v1/swarm.py`
finds the binary at the relative path it was written for (`../uai32/dist/uai32`) without any change to the
frozen code. Set `UAI32=/path/to/uai32` to point it elsewhere.
