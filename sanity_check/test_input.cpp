// Checks the input file parser: globals, chains, defaults, comments, required
// keys and every error case. Needs no GPU.

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

// check_required's message, or "" if it does not throw
std::string missing(const std::string& text) {
  try {
    check_required(parse(text), "test");
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

int main() {
  // a complete file: globals in the three spellings, comments, blank lines
  const std::string complete =
      "# a comment\n"
      "\n"
      "cache = host          # trailing comment\n"
      "muon_mass=105.6583745\n"
      "lattice_spacing 0.07957   # fm\n"
      "out=result.h5\n"
      "timing = t.csv\n"
      "random_fields = on\n"
      "lattice = 6,6,6,6\n"
      "procs = 1 1 1 2\n"
      "chain start=0,0,0,0 dir=1,1,1,1 stride=4 steps=30\n"
      "chain start=32,0,0,0  dir=1,-1,1,1  stride=4 steps=auto\n"
      "chain stride=6 dir=-1,-1,-1,-1 start=0,16,0,-3\n";
  {
    Input in = parse(complete);
    HLBL_CHECK(in.cache == "host");
    HLBL_CHECK(in.muon_mass == 105.6583745);
    HLBL_CHECK(in.lattice_spacing == 0.07957);
    HLBL_CHECK(in.out == "result.h5");
    HLBL_CHECK(in.timing == "t.csv");
    HLBL_CHECK(in.random_fields);
    HLBL_CHECK((*in.lattice == Vec4{6, 6, 6, 6}));
    HLBL_CHECK((*in.procs == Vec4{1, 1, 1, 2}));
    HLBL_CHECK(in.chains.size() == 3);

    const Chain& a = in.chains[0];
    HLBL_CHECK((a.start == Vec4{0, 0, 0, 0}));
    HLBL_CHECK((a.direction == Vec4{1, 1, 1, 1}));
    HLBL_CHECK(a.stride == 4 && a.steps == 30);

    const Chain& b = in.chains[1];
    HLBL_CHECK((b.start == Vec4{32, 0, 0, 0}));
    HLBL_CHECK((b.direction == Vec4{1, -1, 1, 1}));
    HLBL_CHECK(b.steps == -1);  // auto

    const Chain& c = in.chains[2];  // keys in any order; steps defaults to auto
    HLBL_CHECK((c.start == Vec4{0, 16, 0, -3}));
    HLBL_CHECK((c.direction == Vec4{-1, -1, -1, -1}));
    HLBL_CHECK(c.stride == 6 && c.steps == -1);

    check_required(in, "test");  // nothing missing
  }

  // an empty file parses, with every required key missing
  {
    Input in = parse("# nothing\n");
    HLBL_CHECK(in.chains.empty() && !in.lattice && !in.procs);
    HLBL_CHECK(in.cache.empty() && in.out.empty() && in.timing.empty());
    HLBL_CHECK(!in.random_fields);
    HLBL_CHECK(in.muon_mass == 0 && in.lattice_spacing == 0);
    HLBL_CHECK(missing("") ==
               "test: missing chain lines, lattice, procs, muon_mass, "
               "lattice_spacing, out, cache");
  }

  // each required key on its own; timing and random_fields are optional
  {
    auto without = [&](const std::string& prefix) {
      std::string text;
      std::istringstream s(complete);
      for (std::string line; std::getline(s, line);) {
        if (line.rfind(prefix, 0) != 0) text += line + "\n";
      }
      return missing(text);
    };
    HLBL_CHECK(without("chain") == "test: missing chain lines");
    HLBL_CHECK(without("lattice =") == "test: missing lattice");
    HLBL_CHECK(without("procs") == "test: missing procs");
    HLBL_CHECK(without("muon_mass") == "test: missing muon_mass");
    HLBL_CHECK(without("lattice_spacing") == "test: missing lattice_spacing");
    HLBL_CHECK(without("out") == "test: missing out");
    HLBL_CHECK(without("cache") == "test: missing cache");
    HLBL_CHECK(without("timing").empty());
    HLBL_CHECK(without("random_fields").empty());
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
  HLBL_CHECK(fails(ok + " windows=1:3", "unknown chain key 'windows'"));
  HLBL_CHECK(fails(ok + " steps", "key=value"));
  HLBL_CHECK(fails("chain start=0,0,0 dir=1,1,1,1 stride=4", "4 integers"));
  HLBL_CHECK(fails("chain start=0,0,0,0,0 dir=1,1,1,1 stride=4", "4 integers"));
  HLBL_CHECK(fails("chain start=0,0,0,a dir=1,1,1,1 stride=4", "integer"));

  // errors in globals
  HLBL_CHECK(fails("cache = gpu", "device or host"));
  HLBL_CHECK(fails("random_fields = maybe", "yes/no"));
  HLBL_CHECK(fails("lattice = 6,6,6", "4 integers"));
  HLBL_CHECK(fails("colour = red", "unknown option"));
  HLBL_CHECK(fails("chains start=0,0,0,0", "unknown option"));
  HLBL_CHECK(fails("flavors = 2", "unknown option 'flavors'"));
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

    Chain c;
    c.direction = {1, 1, 1, 2};
    c.stride = 4;
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
