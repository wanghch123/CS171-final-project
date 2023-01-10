#ifndef GKXX_WHEEL_TICTOC_HPP
#define GKXX_WHEEL_TICTOC_HPP

#include <chrono>
#include <iostream>
#include <sstream>
#include <type_traits>

namespace gkxx {

struct Clock {
  using clock_type = std::chrono::steady_clock;
  clock_type::time_point start_time;
};

inline Clock tic() {
  return {Clock::clock_type::now()};
}

inline Clock::clock_type::duration toc(const Clock &clock) {
  return Clock::clock_type::now() - clock.start_time;
}

namespace detail {

  template <typename Period>
  constexpr const char *units_suffix() {
    constexpr bool unsupported = !std::is_same<Period, Period>::value;
    static_assert(unsupported, "unsupported period type");
    return "";
  }

#define SPECIALIZE_UNITS_SUFFIX(period, suffix)                                \
  template <>                                                                  \
  constexpr const char *units_suffix<period>() {                               \
    return suffix;                                                             \
  }

  SPECIALIZE_UNITS_SUFFIX(std::atto, "as")
  SPECIALIZE_UNITS_SUFFIX(std::femto, "fs")
  SPECIALIZE_UNITS_SUFFIX(std::pico, "ps")
  SPECIALIZE_UNITS_SUFFIX(std::nano, "ns")
  SPECIALIZE_UNITS_SUFFIX(std::micro, "us")
  SPECIALIZE_UNITS_SUFFIX(std::milli, "ms")
  SPECIALIZE_UNITS_SUFFIX(std::centi, "cs")
  SPECIALIZE_UNITS_SUFFIX(std::deci, "ds")
  SPECIALIZE_UNITS_SUFFIX(std::ratio<1>, "s")
  SPECIALIZE_UNITS_SUFFIX(std::deca, "das")
  SPECIALIZE_UNITS_SUFFIX(std::hecto, "hs")
  SPECIALIZE_UNITS_SUFFIX(std::kilo, "ks")
  SPECIALIZE_UNITS_SUFFIX(std::mega, "Ms")
  SPECIALIZE_UNITS_SUFFIX(std::giga, "Gs")
  SPECIALIZE_UNITS_SUFFIX(std::tera, "Ts")
  SPECIALIZE_UNITS_SUFFIX(std::peta, "Ps")
  SPECIALIZE_UNITS_SUFFIX(std::exa, "Es")
  SPECIALIZE_UNITS_SUFFIX(std::ratio<60>, "min")
  SPECIALIZE_UNITS_SUFFIX(std::ratio<3600>, "h")
  SPECIALIZE_UNITS_SUFFIX(std::ratio<86400>, "d")

#undef SPECIALIZE_UNITS_SUFFIX
} // namespace detail

} // namespace gkxx

template <typename Rep, typename Period>
inline std::ostream &operator<<(std::ostream &os,
                                const std::chrono::duration<Rep, Period> &d) {
  std::ostringstream oss;
  oss.flags(os.flags());
  oss.imbue(os.getloc());
  oss.precision(os.precision());
  using period_type = typename Period::type;
  oss << d.count() << gkxx::detail::units_suffix<period_type>();
  return os << oss.str();
}

#endif // GKXX_WHEEL_TICTOC_HPP