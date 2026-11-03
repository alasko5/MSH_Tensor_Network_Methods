/* Flush-to-zero (FTZ, MXCSR bit 15) and denormals-are-zero (DAZ, bit 6) for the calling thread (x86-64 SSE).
   Used by msh_substrate_python.ipynb, section 5 (method "mumps_ftz"); compiled there into data/libftz.so with
   cc -O2 -shared -fPIC -o data/libftz.so ftz.c */
#include <xmmintrin.h>

/* Switch FTZ and DAZ on (on != 0) or off; returns the previous MXCSR value. */
unsigned int ftz_daz(int on) {
    unsigned int c = _mm_getcsr();
    _mm_setcsr(on ? (c | 0x8040u) : (c & ~0x8040u));
    return c;
}

/* Restore an MXCSR value returned by ftz_daz. */
void restore_csr(unsigned int c) { _mm_setcsr(c); }
