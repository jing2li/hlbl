#pragma once

// Minimal assertion macro for the sanity_check programs: no framework, just
// print-and-exit-nonzero on failure so these run as plain executables or
// CTest cases alike.

#include <cstdio>
#include <cstdlib>

#define HLBL_CHECK(cond)                                                 \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)
