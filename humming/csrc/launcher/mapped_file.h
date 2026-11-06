#pragma once

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <filesystem>
#include <system_error>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

class MappedFile {
public:
  explicit MappedFile(const std::string &path) {
#ifdef _WIN32
    auto wide_path = std::filesystem::u8path(path);
    HANDLE file = CreateFileW(
        wide_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw_error("CreateFileW", path, GetLastError());

    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size)) {
      DWORD error = GetLastError();
      CloseHandle(file);
      throw_error("GetFileSizeEx", path, error);
    }
    if (size.QuadPart <= 0) {
      CloseHandle(file);
      throw std::runtime_error("empty file: " + path);
    }
    size_ = static_cast<size_t>(size.QuadPart);
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    DWORD error = GetLastError();
    CloseHandle(file);
    if (mapping == nullptr) throw_error("CreateFileMappingW", path, error);

    data_ = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    error = GetLastError();
    CloseHandle(mapping);
    if (data_ == nullptr) throw_error("MapViewOfFile", path, error);
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw_error("open", path, errno);

    struct stat st;
    if (fstat(fd, &st) != 0) {
      int error = errno;
      close(fd);
      throw_error("fstat", path, error);
    }
    if (st.st_size <= 0) {
      close(fd);
      throw std::runtime_error("empty file: " + path);
    }

    size_ = static_cast<size_t>(st.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    int error = errno;
    close(fd);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw_error("mmap", path, error);
    }
#endif
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  ~MappedFile() {
#ifdef _WIN32
    if (data_ != nullptr) UnmapViewOfFile(data_);
#else
    if (data_ != nullptr) munmap(data_, size_);
#endif
  }

  const void *data() const { return data_; }
  size_t size() const { return size_; }

private:
  [[noreturn]] static void throw_error(const char *operation, const std::string &path, int error) {
#ifdef _WIN32
    throw std::system_error(error, std::system_category(), std::string(operation) + " failed for " + path);
#else
    throw std::runtime_error(
        std::string(operation) + " failed for " + path + ": " + std::strerror(error));
#endif
  }

  void *data_ = nullptr;
  size_t size_ = 0;
};
