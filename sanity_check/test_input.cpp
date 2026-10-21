// Checks the input file parser: globals, chains, defaults, comments and every
// error case. Needs no GPU.

#include "hlbl/input.hpp"

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

#include "check.hpp"

using namespace hlbl;

namespace {

Input parse(const std::string& text) {
  std::istringstream s(text);
  return parse_input(s, "test");
}

// true if parsing throws and the message contains `where`
bool fails(const std::string& text, const std::string& where = "") {
  try {
    parse(text);
  } catch (const std::runtime_error& e) {
    return std::string(e.what()).find(where) != std::string::npos;
  }
  return false;
}

}  // namespace

int main() {
  // a complete file: globals in the three spellings, comments, blank lines
  {
    Input in = parse(
        "# a comment\n"
        "\n"
        "cache = host          # trailing comment\n"
        "muon_mass=105.6583745\n"
        "lattice_spacing 0.07957   # fm\n"
        "out=result.h5\n"
        "timing = t.csv\n"
        "random_fields = on\n"
        "flavors = 1\n"
        "lattice = 6,6,6,6\n"
        "procs = 1 1 1 2\n"
        "chain start=0,0,0,0 dir=1,1,1,1 stride=4 steps=30\n"
        "chain start=32,0,0,0  dir=1,-1,1,1  stride=4 steps=30 windows=10:20\n"
        "chain stride=6 dir=-1,-1,-1,-1 start=0,16,0,-3\n");
    HLBL_CHECK(!in.device_cache);
    HLBL_CHECK(in.muon_mass == 105.6583745);
    HLBL_CHECK(in.lattice_spacing == 0.07957);
    HLBL_CHECK(in.out == "result.h5");
    HLBL_CHECK(in.timing == "t.csv");
    HLBL_CHECK(in.random_fields);
    HLBL_CHECK(in.flavors == 1);
    HLBL_CHECK((in.lattice == Vec4{6, 6, 6, 6}));
    HLBL_CHECK((in.procs == Vec4{1, 1, 1, 2}));
    HLBL_CHECK(in.chains.size() == 3);

    const Chain& a = in.chains[0];
    HLBL_CHECK((a.start == Vec4{0, 0, 0, 0}));
    HLBL_CHECK((a.direction == Vec4{1, 1, 1, 1}));
    HLBL_CHECK(a.stride == 4 && a.steps == 30);
    HLBL_CHECK(a.window_begin == 0 && a.window_end == -1);  // all windows

    const Chain& b = in.chains[1];
    HLBL_CHECK((b.start == Vec4{32, 0, 0, 0}));
    HLBL_CHECK((b.direction == Vec4{1, -1, 1, 1}));
    HLBL_CHECK(b.window_begin == 10 && b.window_end == 20);

    const Chain& c = in.chains[2];  // keys in any order; steps defaults to auto
    HLBL_CHECK((c.start == Vec4{0, 16, 0, -3}));
    HLBL_CHECK((c.direction == Vec4{-1, -1, -1, -1}));
    HLBL_CHECK(c.stride == 6 && c.steps == -1);
  }

  // an empty file is valid: no chains (the program then uses its defaults)
  {
    Input in = parse("# nothing\n");
    HLBL_CHECK(in.chains.empty());
    HLBL_CHECK(in.device_cache && in.out.empty());
    HLBL_CHECK(in.muon_mass == 0 && in.lattice_spacing == 0);  // not given
  }

  // steps=auto and open window ranges
  {
    Input in = parse(
        "chain start=0,0,0,0 dir=1,1,1,1 stride=2 steps=auto windows=3:\n"
        "chain start=0,0,0,0 dir=1,1,1,1 stride=2 steps=9 windows=:4\n");
    HLBL_CHECK(in.chains[0].steps == -1);
    HLBL_CHECK(in.chains[0].window_begin == 3 && in.chains[0].window_end == -1);
    HLBL_CHECK(in.chains[1].window_begin == 0 && in.chains[1].window_end == 4);
  }

  // errors in chains; the message names the line
  const std::string ok = "chain start=0,0,0,0 dir=1,1,1,1 stride=4";
  HLBL_CHECK(!fails(ok));
  HLBL_CHECK(fails("chain dir=1,1,1,1 stride=4", "start"));
  HLBL_CHECK(fails("chain start=0,0,0,0 stride=4", "dir"));
  HLBL_CHECK(fails("chain start=0,0,0,0 dir=1,1,1,1", "stride"));
  HLBL_CHECK(fails("chain start=0,0,0,0 dir=1,1,1,2 stride=4", "sign vector"));
  HLBL_CHECK(fails("chain start=0,0,0,0 dir=1,1,1,0 stride=4", "sign vector"));
  HLBL_CHECK(fails("chain start=0,0,0,0 dir=1,1,1,1 stride=0", "stride"));
  HLBL_CHECK(fails("chain start=0,0,0,0 dir=1,1,1,1 stride=-4", "stride"));
  HLBL_CHECK(fails(ok + " steps=-2", "steps"));
  HLBL_CHECK(fails(ok + " steps=x", "integer"));
  HLBL_CHECK(fails(ok + " stride=5", "twice"));
  HLBL_CHECK(fails(ok + " colour=1", "unknown chain key"));
  HLBL_CHECK(fails(ok + " steps", "key=value"));
  HLBL_CHECK(fails("chain start=0,0,0 dir=1,1,1,1 stride=4", "4 integers"));
  HLBL_CHECK(fails("chain start=0,0,0,0,0 dir=1,1,1,1 stride=4", "4 integers"));
  HLBL_CHECK(fails("chain start=0,0,0,a dir=1,1,1,1 stride=4", "integer"));
  HLBL_CHECK(fails(ok + " windows=5", "a:b"));
  HLBL_CHECK(fails(ok + " windows=3:3", "empty window range"));
  HLBL_CHECK(fails(ok + " windows=4:2", "empty window range"));
  HLBL_CHECK(fails(ok + " windows=-1:2", "below 0"));
  HLBL_CHECK(fails(ok + " steps=2 windows=3:", "after the last window"));
  HLBL_CHECK(!fails(ok + " steps=2 windows=2:"));

  // errors in globals
  HLBL_CHECK(fails("cache = gpu", "device or host"));
  HLBL_CHECK(fails("random_fields = maybe", "yes/no"));
  HLBL_CHECK(fails("flavors = 3", "1 or 2"));
  HLBL_CHECK(fails("lattice = 6,6,6", "4 integers"));
  HLBL_CHECK(fails("colour = red", "unknown option"));
  HLBL_CHECK(fails("chains start=0,0,0,0", "unknown option"));
  HLBL_CHECK(fails("force = yes", "unknown option"));  // a run checks nothing to force
  HLBL_CHECK(fails("muon_mass = 0", "muon_mass must be a positive number (MeV)"));
  HLBL_CHECK(fails("lattice_spacing = -0.08", "positive"));
  HLBL_CHECK(fails("muon_mass = 1e999", "positive"));
  HLBL_CHECK(fails("lattice_spacing = abc", "not a number"));
  HLBL_CHECK(fails("lattice_spacing = inf", "not a number"));
  HLBL_CHECK(fails("muon_mass = 105.66 MeV", "not a number"));  // units go in a comment

  // the message names the file and the line
  HLBL_CHECK(fails("\n# comment\nchain start=0,0,0,0 dir=1,1,1,3 stride=4\n", "test:3:"));
  HLBL_CHECK(fails("cache = host\ncache = gpu\n", "test:2:"));

  // helpers
  {
    HLBL_CHECK((parse_vec4("1, 2,3 4") == Vec4{1, 2, 3, 4}));
    HLBL_CHECK(parse_positive(" 7.957e-2 ", "a", "fm") == 0.07957);
    int b = -1, e = -1;
    parse_window_range("2:5", b, e);
    HLBL_CHECK(b == 2 && e == 5);
    parse_window_range("2:", b, e);
    HLBL_CHECK(b == 2 && e == -1);

    Chain c;  // the command-line defaults are a valid chain
    validate(c);
    c.direction = {1, 1, 1, 2};
    bool threw = false;
    try {
      validate(c);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    HLBL_CHECK(threw);

    threw = false;
    try {
      parse_input_file("/nonexistent/hlbl.input");
    } catch (const std::runtime_error&) {
      threw = true;
    }
    HLBL_CHECK(threw);
  }

  std::printf("test_input: all checks passed\n");
  return 0;
}
