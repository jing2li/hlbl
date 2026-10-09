#pragma once

// Minimal assertion macro for the tests: prints and exits with an error code.

#include <cstdio>
#include <cstdlib>

#define HLBL_CHECK(cond)                                                 \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)
