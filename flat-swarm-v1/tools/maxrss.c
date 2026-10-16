/* maxrss.c -- run a command and print its peak resident set size (ru_maxrss, KB) and wall time.
 *   cc -O2 maxrss.c -o maxrss;  ./maxrss [-i STDIN_FILE] CMD ARGS...
 * Uses posix_spawn, not fork: a forked child shares the parent's address space until exec, and the
 * kernel charges those pages to the child's ru_maxrss, so measuring from a Python/Bash parent reports the
 * parent's RSS (about 11 MB from python3) for every command, even /bin/true.  posix_spawn (vfork-like)
 * avoids that; the number printed is the command's own peak RSS plus the kernel's ELF loader floor. */
#define _GNU_SOURCE
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <time.h>
extern char **environ;
int main(int argc, char **argv) {
    posix_spawn_file_actions_t fa; pid_t pid; int st; struct rusage ru; struct timespec t0, t1;
    const char *in = NULL;
    if (argc > 2 && !strcmp(argv[1], "-i")) { in = argv[2]; argv += 2; argc -= 2; }
    if (argc < 2) { fprintf(stderr, "usage: maxrss [-i STDIN_FILE] CMD ARGS...\n"); return 2; }
    posix_spawn_file_actions_init(&fa);
    if (in) posix_spawn_file_actions_addopen(&fa, 0, in, O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (posix_spawn(&pid, argv[1], &fa, NULL, argv + 1, environ)) { perror("posix_spawn"); return 2; }
    if (wait4(pid, &st, 0, &ru) < 0) { perror("wait4"); return 2; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("%s: maxrss %ld KB, wall %.2f ms, exit %d\n", argv[1], ru.ru_maxrss,
           ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1e3, WEXITSTATUS(st));
    return 0;
}
