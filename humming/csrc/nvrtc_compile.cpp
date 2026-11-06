#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <system_error>
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <nvrtc.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

decltype(&::nvrtcAddNameExpression) p_nvrtcAddNameExpression;
decltype(&::nvrtcCompileProgram) p_nvrtcCompileProgram;
decltype(&::nvrtcCreateProgram) p_nvrtcCreateProgram;
decltype(&::nvrtcDestroyProgram) p_nvrtcDestroyProgram;
decltype(&::nvrtcGetCUBIN) p_nvrtcGetCUBIN;
decltype(&::nvrtcGetCUBINSize) p_nvrtcGetCUBINSize;
decltype(&::nvrtcGetErrorString) p_nvrtcGetErrorString;
decltype(&::nvrtcGetPTX) p_nvrtcGetPTX;
decltype(&::nvrtcGetPTXSize) p_nvrtcGetPTXSize;
decltype(&::nvrtcGetProgramLog) p_nvrtcGetProgramLog;
decltype(&::nvrtcGetProgramLogSize) p_nvrtcGetProgramLogSize;

#define nvrtcAddNameExpression p_nvrtcAddNameExpression
#define nvrtcCompileProgram p_nvrtcCompileProgram
#define nvrtcCreateProgram p_nvrtcCreateProgram
#define nvrtcDestroyProgram p_nvrtcDestroyProgram
#define nvrtcGetCUBIN p_nvrtcGetCUBIN
#define nvrtcGetCUBINSize p_nvrtcGetCUBINSize
#define nvrtcGetErrorString p_nvrtcGetErrorString
#define nvrtcGetPTX p_nvrtcGetPTX
#define nvrtcGetPTXSize p_nvrtcGetPTXSize
#define nvrtcGetProgramLog p_nvrtcGetProgramLog
#define nvrtcGetProgramLogSize p_nvrtcGetProgramLogSize

void die(const std::string &msg) {
  std::fprintf(stderr, "nvrtc_compile: %s\n", msg.c_str());
  std::exit(2);
}

#ifdef _WIN32
using LibraryHandle = HMODULE;

std::string to_utf8(const wchar_t *value) {
  int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, nullptr, 0, nullptr, nullptr);
  if (size == 0)
    die("cannot encode argument as UTF-8: " + std::system_category().message(GetLastError()));
  std::string result(size, '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, result.data(), size, nullptr, nullptr) == 0)
    die("cannot encode argument as UTF-8: " + std::system_category().message(GetLastError()));
  result.pop_back();
  return result;
}
#else
using LibraryHandle = void *;
#endif

template <typename T>
void load_symbol(LibraryHandle library, T &symbol, const char *name) {
#ifdef _WIN32
  symbol = reinterpret_cast<T>(GetProcAddress(library, name));
  if (symbol == nullptr)
    die(std::string("cannot load ") + name + ": " + std::system_category().message(GetLastError()));
#else
  symbol = reinterpret_cast<T>(dlsym(library, name));
  if (symbol == nullptr) die(std::string("cannot load ") + name + ": " + dlerror());
#endif
}

void load_nvrtc(const std::string &configured_path) {
#ifdef _WIN32
  auto wide_path = std::filesystem::absolute(std::filesystem::u8path(configured_path));
  // NVRTC loads its builtins later, so keep this directory available for the process lifetime.
  if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS))
    die("cannot configure DLL search: " + std::system_category().message(GetLastError()));
  if (AddDllDirectory(wide_path.parent_path().c_str()) == nullptr)
    die("cannot add NVRTC DLL directory: " + std::system_category().message(GetLastError()));
  LibraryHandle library = LoadLibraryExW(
      wide_path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (library == nullptr)
    die("cannot load NVRTC from " + configured_path + ": " + std::system_category().message(GetLastError()));
#else
  void *library = dlopen(configured_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr)
    die("cannot load NVRTC from " + configured_path + ": " + dlerror());
#endif
  load_symbol(library, p_nvrtcAddNameExpression, "nvrtcAddNameExpression");
  load_symbol(library, p_nvrtcCompileProgram, "nvrtcCompileProgram");
  load_symbol(library, p_nvrtcCreateProgram, "nvrtcCreateProgram");
  load_symbol(library, p_nvrtcDestroyProgram, "nvrtcDestroyProgram");
  load_symbol(library, p_nvrtcGetCUBIN, "nvrtcGetCUBIN");
  load_symbol(library, p_nvrtcGetCUBINSize, "nvrtcGetCUBINSize");
  load_symbol(library, p_nvrtcGetErrorString, "nvrtcGetErrorString");
  load_symbol(library, p_nvrtcGetPTX, "nvrtcGetPTX");
  load_symbol(library, p_nvrtcGetPTXSize, "nvrtcGetPTXSize");
  load_symbol(library, p_nvrtcGetProgramLog, "nvrtcGetProgramLog");
  load_symbol(library, p_nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
}

std::string read_file(const std::string &path) {
  std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) die("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void write_file(const std::string &path, const void *data, size_t size) {
  std::ofstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) die("cannot write " + path);
  f.write(static_cast<const char *>(data), size);
}

#define NVRTC_CHECK(call)                                            \
  do {                                                               \
    nvrtcResult _r = (call);                                         \
    if (_r != NVRTC_SUCCESS) {                                       \
      die(std::string(#call " failed: ") + nvrtcGetErrorString(_r)); \
    }                                                                \
  } while (0)

struct Args {
  std::string nvrtc_path;
  std::string input;
  std::string output;
  std::string log_path;
  bool emit_ptx = false;
  std::vector<std::string> name_exprs;
  std::vector<std::pair<std::string, std::string>> headers; // name -> file path
  std::vector<std::string> nvrtc_flags;
};

Args parse_args(const std::vector<std::string> &argv) {
  Args a;
  size_t i = 1;
  auto need = [&](const char *opt) {
    if (i + 1 >= argv.size()) die(std::string("missing value for ") + opt);
    return argv[++i];
  };
  for (; i < argv.size(); ++i) {
    std::string s = argv[i];
    if (s == "--") {
      ++i;
      break;
    } else if (s == "--nvrtc-path") a.nvrtc_path = need("--nvrtc-path");
    else if (s == "-i" || s == "--input") a.input = need("--input");
    else if (s == "-o" || s == "--output") a.output = need("--output");
    else if (s == "--log") a.log_path = need("--log");
    else if (s == "--ptx") a.emit_ptx = true;
    else if (s == "--name-expression") a.name_exprs.push_back(need("--name-expression"));
    else if (s == "--header") {
      std::string v = need("--header");
      auto eq = v.find('=');
      if (eq == std::string::npos) die("--header expects NAME=PATH");
      a.headers.emplace_back(v.substr(0, eq), v.substr(eq + 1));
    } else if (s == "-h" || s == "--help") {
      std::printf(
          "Usage: nvrtc_compile --input X.cu --output Y.cubin "
          "[--nvrtc-path PATH] [--ptx] [--log path] "
          "[--name-expression EXPR ...] [--header NAME=PATH ...] "
          "-- <nvrtc flag> ...\n");
      std::exit(0);
    } else die("unknown option: " + s);
  }
  for (; i < argv.size(); ++i)
    a.nvrtc_flags.emplace_back(argv[i]);
  if (a.nvrtc_path.empty()) die("--nvrtc-path is required");
  if (a.input.empty()) die("--input is required");
  if (a.output.empty()) die("--output is required");
  return a;
}

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t **argv) {
  std::vector<std::string> arguments;
  arguments.reserve(argc);
  for (int i = 0; i < argc; ++i)
    arguments.push_back(to_utf8(argv[i]));
#else
int main(int argc, char **argv) {
  std::vector<std::string> arguments(argv, argv + argc);
#endif
  Args args = parse_args(arguments);
  load_nvrtc(args.nvrtc_path);

  std::string src = read_file(args.input);

  std::vector<std::string> header_contents;
  header_contents.reserve(args.headers.size());
  std::vector<const char *> header_names_c;
  std::vector<const char *> header_sources_c;
  header_names_c.reserve(args.headers.size());
  header_sources_c.reserve(args.headers.size());
  for (auto &[name, path] : args.headers) {
    header_contents.push_back(read_file(path));
    header_names_c.push_back(name.c_str());
    header_sources_c.push_back(header_contents.back().c_str());
  }

  const char *program_name = args.input.c_str();
#ifdef _WIN32
  // NVRTC's in-memory source lookup rejects non-ASCII program names on Windows.
  bool has_non_ascii_name = std::any_of(args.input.begin(), args.input.end(), [](unsigned char c) {
    return c >= 0x80;
  });
  if (has_non_ascii_name) program_name = nullptr;
#endif
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(
      &prog, src.c_str(), program_name,
      static_cast<int>(args.headers.size()),
      header_sources_c.empty() ? nullptr : header_sources_c.data(),
      header_names_c.empty() ? nullptr : header_names_c.data()));

  for (const auto &e : args.name_exprs) {
    NVRTC_CHECK(nvrtcAddNameExpression(prog, e.c_str()));
  }

  std::vector<const char *> opts;
  opts.reserve(args.nvrtc_flags.size());
  for (const auto &f : args.nvrtc_flags)
    opts.push_back(f.c_str());

  nvrtcResult compile_r = nvrtcCompileProgram(
      prog, static_cast<int>(opts.size()), opts.empty() ? nullptr : opts.data());

  size_t log_size = 0;
  nvrtcGetProgramLogSize(prog, &log_size);
  std::string log(log_size, '\0');
  if (log_size > 0) nvrtcGetProgramLog(prog, log.data());
  if (!log.empty() && log.back() == '\0') log.pop_back();

  if (!log.empty()) std::fwrite(log.data(), 1, log.size(), stderr);
  if (!args.log_path.empty()) write_file(args.log_path, log.data(), log.size());

  if (compile_r != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    std::fprintf(stderr, "\nnvrtc_compile: compile failed for %s: %s\n",
        args.input.c_str(), nvrtcGetErrorString(compile_r));
    return 1;
  }

  size_t out_size = 0;
  if (args.emit_ptx) {
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &out_size));
    std::string buf(out_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, buf.data()));
    write_file(args.output, buf.data(), out_size);
  } else {
    NVRTC_CHECK(nvrtcGetCUBINSize(prog, &out_size));
    if (out_size == 0) {
      nvrtcDestroyProgram(&prog);
      std::fprintf(stderr, "nvrtc_compile: empty CUBIN "
                           "(missing --gpu-architecture=sm_XX?)\n");
      return 1;
    }
    std::string buf(out_size, '\0');
    NVRTC_CHECK(nvrtcGetCUBIN(prog, buf.data()));
    write_file(args.output, buf.data(), out_size);
  }

  nvrtcDestroyProgram(&prog);
  return 0;
}
