/*
 * mono_bridge.c — Minimal LD_PRELOAD library for Cuphead/Wine.
 *
 * This does ONLY three things:
 *   1. Calls prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) so external processes
 *      can use process_vm_readv() to read our memory.
 *   2. Finds mono.dll's base address in /proc/self/maps.
 *   3. Writes PID + mono.dll base + game process info to /dev/shm/cuphead_state.
 *
 * All actual memory reading (chasing Mono pointers, reading HP, etc.) is done
 * by Python externally via process_vm_readv.  This avoids the nightmare of
 * calling PE functions from ELF code (TEB issues, calling convention, etc.)
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -Wall -o core/mono_bridge.so core/mono_bridge.c -lrt
 *
 * Usage:
 *   LD_PRELOAD=./core/mono_bridge.so wine game/CupHead/Cuphead.exe
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <errno.h>
#include <stdarg.h>

/* --------------------------------------------------------------------------
 * Shared memory layout — Python reads this from /dev/shm/cuphead_state
 * -------------------------------------------------------------------------- */

#define SHM_NAME "/cuphead_state"
#define SHM_SIZE 4096
#define MAGIC 0x43555048  /* "CUPH" */

/*
 * Minimal shared state.  Python uses this to bootstrap process_vm_readv:
 *   - pid:       target PID for process_vm_readv
 *   - mono_base: base address of mono.dll in process memory
 *   - mono_size: total mapped size of mono.dll
 *
 * Python then parses mono.dll's PE export table via process_vm_readv,
 * resolves Mono API function pointers, reads root domain, walks the
 * class hierarchy, and reads game state — all externally.
 *
 * Well, actually Python can't CALL the Mono API functions via
 * process_vm_readv (it's read-only). But it CAN:
 *   - Find static field data by walking Mono metadata structures
 *   - Read the values of those fields directly from memory
 *   - Chase pointer chains: PlayerManager → players → _stats → HP
 */
struct bridge_state {
    uint32_t magic;           /* 0x00: MAGIC to verify valid data */
    uint32_t version;         /* 0x04: struct version */
    uint32_t pid;             /* 0x08: game process PID */
    uint32_t flags;           /* 0x0C: status flags */
    uint64_t mono_base;       /* 0x10: mono.dll base address */
    uint64_t mono_size;       /* 0x18: mono.dll mapped size */
    uint64_t timestamp_ns;    /* 0x20: when last updated */

    /* Bridge status */
    int32_t  bridge_state;    /* 0x28: 0=init, 1=scanning, 2=ready, -1=error */
    char     error_msg[128];  /* 0x2C: last error message */

    /* Debug log */
    uint32_t debug_log_len;   /* 0xAC: current length of debug log */
    char     debug_log[3408]; /* 0xB0: debug text (fills rest of 4KB page) */
};

#define FLAG_PRCTL_OK    (1 << 0)
#define FLAG_MONO_FOUND  (1 << 1)
#define FLAG_READY       (1 << 2)

/* --------------------------------------------------------------------------
 * Global state
 * -------------------------------------------------------------------------- */

static struct {
    struct bridge_state *shm;
    int shm_fd;
    volatile int running;
    pthread_t thread;
} g = {0};

/* Log to stderr and shared memory debug log */
#define LOG(fmt, ...) do { \
    fprintf(stderr, "[mono_bridge] " fmt "\n", ##__VA_ARGS__); \
    debug_append(fmt "\n", ##__VA_ARGS__); \
} while(0)

static void debug_append(const char *fmt, ...)
    __attribute__((format(printf, 1, 2)));
static void debug_append(const char *fmt, ...)
{
    if (!g.shm) return;
    va_list ap;
    va_start(ap, fmt);
    uint32_t used = g.shm->debug_log_len;
    uint32_t avail = sizeof(g.shm->debug_log) - 1 - used;
    if (avail > 0) {
        int n = vsnprintf(g.shm->debug_log + used, avail, fmt, ap);
        if (n > 0) {
            g.shm->debug_log_len = used + ((uint32_t)n < avail ? (uint32_t)n : avail - 1);
        }
    }
    va_end(ap);
}

/* --------------------------------------------------------------------------
 * Find mono.dll in /proc/self/maps
 * -------------------------------------------------------------------------- */

static int find_mono_base(uint64_t *out_base, uint64_t *out_size)
{
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return -1;

    char line[512];
    uint64_t mono_start = 0, mono_end = 0;

    while (fgets(line, sizeof(line), f)) {
        uint64_t seg_start, seg_end;
        if (sscanf(line, "%lx-%lx", &seg_start, &seg_end) != 2)
            continue;

        if (strstr(line, "mono.dll")) {
            if (!mono_start) mono_start = seg_start;
            if (seg_end > mono_end) mono_end = seg_end;
        } else if (mono_start && seg_start == mono_end) {
            /* Contiguous anonymous region — PE section */
            mono_end = seg_end;
        } else if (mono_start && seg_start > mono_end) {
            break;
        }
    }
    fclose(f);

    if (!mono_start) return -1;

    *out_base = mono_start;
    *out_size = mono_end - mono_start;
    return 0;
}

/* --------------------------------------------------------------------------
 * Shared memory setup
 * -------------------------------------------------------------------------- */

static int setup_shm(void)
{
    shm_unlink(SHM_NAME);

    g.shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0644);
    if (g.shm_fd < 0) return -1;

    ftruncate(g.shm_fd, SHM_SIZE);

    g.shm = (struct bridge_state *)mmap(NULL, SHM_SIZE,
        PROT_READ | PROT_WRITE, MAP_SHARED, g.shm_fd, 0);
    if (g.shm == MAP_FAILED) {
        close(g.shm_fd);
        return -1;
    }

    memset(g.shm, 0, SHM_SIZE);
    g.shm->magic = MAGIC;
    g.shm->version = 2;  /* v2 = external reader */
    g.shm->pid = (uint32_t)getpid();
    g.shm->bridge_state = 0;

    return 0;
}

/* --------------------------------------------------------------------------
 * Background thread — polls for mono.dll, updates shm
 * -------------------------------------------------------------------------- */

static void *scanner_thread(void *arg)
{
    (void)arg;
    LOG("Scanner thread started (tid=%d)", (int)syscall(SYS_gettid));

    g.shm->bridge_state = 1;  /* scanning */

    /* Poll for mono.dll */
    uint64_t mono_base = 0, mono_size = 0;
    for (int i = 0; i < 120; i++) {  /* 120 * 500ms = 60s */
        if (find_mono_base(&mono_base, &mono_size) == 0)
            break;
        usleep(500000);
        if (!g.running) return NULL;
    }

    if (!mono_base) {
        LOG("FATAL: mono.dll not found after 60s");
        snprintf(g.shm->error_msg, sizeof(g.shm->error_msg),
                 "mono.dll not found");
        g.shm->bridge_state = -1;
        return NULL;
    }

    g.shm->mono_base = mono_base;
    g.shm->mono_size = mono_size;
    g.shm->flags |= FLAG_MONO_FOUND;

    LOG("mono.dll: base=0x%lx size=%lu (%lu KB)",
        mono_base, mono_size, mono_size / 1024);

    /* Update timestamp and mark ready */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    g.shm->timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    g.shm->flags |= FLAG_READY;
    g.shm->bridge_state = 2;  /* ready */

    LOG("=== Bridge READY ===");
    LOG("  PID:       %d", g.shm->pid);
    LOG("  mono.dll:  0x%lx (%lu KB)", mono_base, mono_size / 1024);
    LOG("  prctl:     %s", (g.shm->flags & FLAG_PRCTL_OK) ? "OK" : "FAILED");
    LOG("  Python can now read game state via process_vm_readv");
    LOG("  Run: python -m core.game_state_reader");

    /* Keep the thread alive (shm stays mapped) — just heartbeat */
    while (g.running) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        g.shm->timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        sleep(1);
    }

    return NULL;
}

/* --------------------------------------------------------------------------
 * Constructor / Destructor
 * -------------------------------------------------------------------------- */

static int is_game_process(void)
{
    char buf[4096];
    FILE *f = fopen("/proc/self/cmdline", "r");
    if (!f) return 0;

    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';

    for (size_t i = 0; i < n; i++) {
        if (buf[i] == '\0') buf[i] = ' ';
    }

    return (strcasestr(buf, "Cuphead.exe") != NULL);
}

static int g_is_active = 0;

__attribute__((constructor))
static void bridge_init(void)
{
    if (!is_game_process()) return;

    fprintf(stderr, "[mono_bridge] === Mono Bridge v4 (external reader) loaded ===\n");
    g_is_active = 1;

    if (setup_shm() < 0) {
        fprintf(stderr, "[mono_bridge] Failed to setup shared memory\n");
        g_is_active = 0;
        return;
    }

    LOG("PID: %d", getpid());

    /*
     * THE KEY CALL: allow any same-UID process to read our memory.
     *
     * Wine intercepts syscalls via its preloader, so libc's prctl()
     * may go through Wine's syscall dispatcher and never reach the kernel.
     * We use inline assembly to make a direct syscall instruction.
     *
     * SYS_prctl = 157
     * PR_SET_PTRACER = 0x59616d61
     * PR_SET_PTRACER_ANY = (unsigned long)-1
     */
    long prctl_ret;
    __asm__ volatile(
        "mov $157, %%rax\n"         /* SYS_prctl */
        "mov $0x59616d61, %%rdi\n"  /* PR_SET_PTRACER */
        "mov $-1, %%rsi\n"          /* PR_SET_PTRACER_ANY */
        "xor %%rdx, %%rdx\n"
        "xor %%r10, %%r10\n"
        "xor %%r8, %%r8\n"
        "syscall\n"
        "mov %%rax, %0\n"
        : "=r"(prctl_ret)
        :
        : "rax", "rdi", "rsi", "rdx", "r10", "r8", "rcx", "r11", "memory"
    );

    if (prctl_ret == 0) {
        LOG("prctl(PR_SET_PTRACER, ANY) OK via direct syscall");
        g.shm->flags |= FLAG_PRCTL_OK;
    } else {
        LOG("prctl direct syscall failed: ret=%ld", prctl_ret);
        /* Also try the libc wrapper as fallback */
        if (prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0) == 0) {
            LOG("prctl via libc wrapper OK (fallback)");
            g.shm->flags |= FLAG_PRCTL_OK;
        } else {
            LOG("prctl completely failed — external reads won't work");
        }
    }

    g.running = 1;

    /* Spawn scanner thread to find mono.dll (may not be loaded yet) */
    if (pthread_create(&g.thread, NULL, scanner_thread, NULL) != 0) {
        LOG("Failed to create scanner thread");
        g_is_active = 0;
        return;
    }
    LOG("Scanner thread launched");
}

__attribute__((destructor))
static void bridge_cleanup(void)
{
    if (!g_is_active) return;

    LOG("=== Mono Bridge unloading ===");
    g.running = 0;
    usleep(100000);

    if (g.shm) {
        munmap(g.shm, SHM_SIZE);
    }
    if (g.shm_fd >= 0) {
        close(g.shm_fd);
        shm_unlink(SHM_NAME);
    }
}
