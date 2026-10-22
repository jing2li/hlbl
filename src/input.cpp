#include "hlbl/input.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace hlbl {

namespace {

std::string trim(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

int parse_int(const std::string& text) {
  std::string const t = trim(text);
  char* end = nullptr;
  long const v = std::strtol(t.c_str(), &end, 10);
  if (t.empty() || *end != '\0') {
    throw std::runtime_error("not an integer: '" + t + "'");
  }
  return static_cast<int>(v);
}

// Decimal notation only (digits, sign, point, exponent): strtod would also take
// inf, nan and hex floats, which the Python reference does not.
double parse_double(const std::string& text) {
  std::string const t = trim(text);
  bool const decimal =
      !t.empty() && t.find_first_not_of("0123456789+-.eE") == std::string::npos;
  char* end = nullptr;
  double const v = decimal ? std::strtod(t.c_str(), &end) : 0.0;
  if (!decimal || *end != '\0') {
    throw std::runtime_error("not a number: '" + t + "'");
  }
  return v;
}

bool parse_bool(const std::string& text) {
  std::string t = trim(text);
  for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (t == "yes" || t == "on" || t == "true" || t == "1") return true;
  if (t == "no" || t == "off" || t == "false" || t == "0") return false;
  throw std::runtime_error("not a yes/no value: '" + trim(text) + "'");
}

// Sets one global option; returns false for an unknown key.
bool set_global(Input& in, const std::string& key, const std::string& value) {
  if (key == "cache") {
    std::string const v = trim(value);
    if (v != "device" && v != "host") {
      throw std::runtime_error("cache must be device or host");
    }
    in.cache = v;
  } else if (key == "random_fields") {
    in.random_fields = parse_bool(value);
  } else if (key == "out") {
    in.out = trim(value);
  } else if (key == "timing") {
    in.timing = trim(value);
  } else if (key == "lattice") {
    in.lattice = parse_vec4(trim(value));
  } else if (key == "procs") {
    in.procs = parse_vec4(trim(value));
  } else if (key == "muon_mass") {
    in.muon_mass = parse_positive(value, key, "MeV");
  } else if (key == "lattice_spacing") {
    in.lattice_spacing = parse_positive(value, key, "fm");
  } else {
    return false;
  }
  return true;
}

// `chain key=value ...`; `tokens` are the words after "chain".
Chain parse_chain(const std::vector<std::string>& tokens) {
  Chain c;
  std::set<std::string> seen;
  for (const std::string& token : tokens) {
    auto const eq = token.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error("expected key=value, got '" + token + "'");
    }
    std::string const key = token.substr(0, eq), value = token.substr(eq + 1);
    if (!seen.insert(key).second) {
      throw std::runtime_error("chain key '" + key + "' given twice");
    }
    if (key == "start") {
      c.start = parse_vec4(value);
    } else if (key == "dir") {
      c.direction = parse_vec4(value);
    } else if (key == "stride") {
      c.stride = parse_int(value);
    } else if (key == "steps") {
      c.steps = value == "auto" ? -1 : parse_int(value);
    } else {
      throw std::runtime_error("unknown chain key '" + key + "'");
    }
  }
  for (const char* required : {"start", "dir", "stride"}) {
    if (!seen.count(required)) {
      throw std::runtime_error(std::string("chain needs ") + required + "=");
    }
  }
  validate(c);
  return c;
}

}  // namespace

Vec4 parse_vec4(const std::string& text) {
  std::string t = text;
  for (char& c : t) {
    if (c == ',') c = ' ';
  }
  std::istringstream s(t);
  Vec4 v{};
  std::string word;
  for (int k = 0; k < 4; ++k) {
    if (!(s >> word)) throw std::runtime_error("expected 4 integers: '" + text + "'");
    v[k] = parse_int(word);
  }
  if (s >> word) throw std::runtime_error("expected 4 integers: '" + text + "'");
  return v;
}

double parse_positive(const std::string& text, const std::string& name,
                      const std::string& unit) {
  double const v = parse_double(text);
  if (!std::isfinite(v) || v <= 0) {
    throw std::runtime_error(name + " must be a positive number (" + unit + ")");
  }
  return v;
}

void validate(const Chain& c) {
  for (int d : c.direction) {
    if (d != 1 && d != -1) {
      throw std::runtime_error("dir must be a sign vector (+-1,+-1,+-1,+-1)");
    }
  }
  if (c.stride <= 0) throw std::runtime_error("stride must be positive");
  if (c.steps < -1) throw std::runtime_error("steps must be >= 0 (or auto)");
}

void check_required(const Input& in, const std::string& name) {
  std::vector<std::string> missing;
  if (in.chains.empty()) missing.push_back("chain lines");
  if (!in.lattice) missing.push_back("lattice");
  if (!in.procs) missing.push_back("procs");
  if (in.muon_mass <= 0) missing.push_back("muon_mass");
  if (in.lattice_spacing <= 0) missing.push_back("lattice_spacing");
  if (in.out.empty()) missing.push_back("out");
  if (in.cache.empty()) missing.push_back("cache");
  if (missing.empty()) return;
  std::string list;
  for (const std::string& m : missing) list += (list.empty() ? "" : ", ") + m;
  throw std::runtime_error(name + ": missing " + list);
}

Input parse_input(std::istream& in, const std::string& name) {
  Input input;
  std::string line;
  int number = 0;
  while (std::getline(in, line)) {
    ++number;
    auto const hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    line = trim(line);
    if (line.empty()) continue;
    try {
      std::istringstream words(line);
      std::string first;
      words >> first;
      if (first == "chain") {
        std::vector<std::string> tokens;
        for (std::string w; words >> w;) tokens.push_back(w);
        input.chains.push_back(parse_chain(tokens));
        continue;
      }
      // `key = value`, `key=value` or `key value`
      auto const eq = line.find('=');
      std::string key, value;
      if (eq != std::string::npos) {
        key = trim(line.substr(0, eq));
        value = line.substr(eq + 1);
      } else {
        key = first;
        value = line.substr(first.size());
      }
      if (!set_global(input, key, value)) {
        throw std::runtime_error("unknown option '" + key + "'");
      }
    } catch (const std::exception& e) {
      throw std::runtime_error(name + ":" + std::to_string(number) + ": " + e.what());
    }
  }
  return input;
}

Input parse_input_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open input file " + path);
  return parse_input(in, path);
}

}  // namespace hlbl
