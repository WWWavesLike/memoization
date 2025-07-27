#pragma once
#include <algorithm>
#include <any>
#include <concepts>
#include <cstddef>
#include <expected>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

namespace opt_utils {

template <typename T>
concept is_raw_ptr_v = std::is_pointer_v<std::remove_cvref_t<T>>;

template <template <typename...> class U, typename T>
struct is_specialization_of : std::false_type {};

template <template <typename...> class U, typename... Args>
struct is_specialization_of<U, U<Args...>> : std::true_type {};

template <template <typename...> class U, typename T>
inline constexpr bool is_specialization_of_v =
	is_specialization_of<U, T>::value;

template <typename T>
concept is_smart_ptr_v =
	is_specialization_of<std::unique_ptr, std::remove_cvref_t<T>>::value ||
	is_specialization_of<std::shared_ptr, std::remove_cvref_t<T>>::value ||
	is_specialization_of<std::weak_ptr, std::remove_cvref_t<T>>::value;
template <typename T>
concept has_lvalue_reference_v = std::is_lvalue_reference_v<T>;

template <typename T>
concept has_ref_or_ptr_v =
	is_raw_ptr_v<T> ||
	is_smart_ptr_v<T> ||
	has_lvalue_reference_v<T>;

template <typename T>
concept tuple_like = requires {
	typename std::tuple_size<T>::type;
};
template <typename T>
concept variant_like = is_specialization_of<std::variant, std::remove_cvref_t<T>>::value;

template <typename T>
concept optional_like = is_specialization_of<std::optional, std::remove_cvref_t<T>>::value;

template <typename T>
concept has_ref_or_ptr_in_container =
	(std::same_as<std::remove_cvref_t<T>, std::any>) ||
	/*
		(is_specialization_of<std::excepted, std::remove_cvref_t<T>>::value &&
		 has_ref_or_ptr_v<typename std::remove_cvref_t<T>::value_type>) ||
	*/
	(optional_like<T> &&
	 has_ref_or_ptr_v<typename std::remove_cvref_t<T>::value_type>) ||
	(variant_like<T> && ([]<std::size_t... Is>(std::index_sequence<Is...>) constexpr {
		 return (has_ref_or_ptr_v<std::variant_alternative_t<Is, T>> || ...);
	 }(std::make_index_sequence<std::variant_size_v<T>>{}))) ||
	(tuple_like<T> &&
	 ([]<std::size_t... Is>(std::index_sequence<Is...>) constexpr {
		 return (has_ref_or_ptr_v<std::tuple_element_t<Is, T>> || ...);
	 }(std::make_index_sequence<std::tuple_size_v<T>>{})));

template <typename T>
concept is_forbidden_wrapper_v =
	(is_specialization_of<std::variant, std::remove_cvref_t<T>>::value &&
	 has_ref_or_ptr_in_container<T>) ||
	(is_specialization_of<std::optional, std::remove_cvref_t<T>>::value &&
	 has_ref_or_ptr_in_container<T>) ||
	std::same_as<std::remove_cvref_t<T>, std::any>;

template <typename R>
concept deterministic =
	(!has_ref_or_ptr_v<std::remove_cvref_t<R>>) &&
	(!has_ref_or_ptr_in_container<R>) &&
	std::equality_comparable<std::remove_cvref_t<R>>;

/*
 *
 ****************************************************************
 *
 */

// containers
struct ordered {};
struct unordered {};

template <typename Tag, typename K, typename V>
struct container_of;

template <typename K, typename V>
struct container_of<ordered, K, V> {
	using type = std::map<K, V>;
};

template <typename K, typename V>
struct container_of<unordered, K, V> {
	using type = std::unordered_map<K, V>;
};

template <typename Tag, typename K, typename V>
using container_t = typename container_of<Tag, K, V>::type;

template <typename P>
concept container_policy = std::same_as<P, ordered> ||
						   std::same_as<P, unordered>;

// limit
template <std::size_t Size>
struct limited {};

template <typename T>
struct is_limited : std::false_type {};

template <std::size_t Size>
struct is_limited<limited<Size>> : std::true_type {};

struct unlimited {};

template <typename Limit, typename T>
struct sub_container {};

template <std::size_t Size, typename T>
struct sub_container<limited<Size>, T> {
	using type = typename T::iterator;
	static constexpr std::size_t size = Size;
	std::list<type> insertion_order;
};

template <typename L>
concept limit_policy = std::same_as<L, unlimited> ||
					   is_limited<L>::value;
/*************************************************/
template <typename T>
struct function_traits; // 일반 템플릿 선언

// 1-1) 함수 포인터
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
	using signature = R(Args...);
};

// 1-2) std::function
template <typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> {
	using signature = R(Args...);
};

// 1-3) 멤버 함수 포인터 (functor의 operator() 특수화용)
template <typename c, typename r, typename... args>
struct function_traits<r (c::*)(args...) const> {
	using signature = r(args...);
};
template <typename c, typename r, typename... args>
struct function_traits<r (c::*)(args...)> {
	using signature = r(args...);
};
// 1-4) 그 외 functor & 람다
template <typename F>
struct function_traits {
	using signature = typename function_traits<
		decltype(&std::decay_t<F>::operator())>::signature;
};

// basic template
template <typename Signature, typename Order = ordered, typename Limit = unlimited>
class memoization;

// specialization
template <typename R, typename... Args, typename Order, typename Limit>
	requires deterministic<R> && container_policy<Order> && limit_policy<Limit>
class memoization<R(Args...), Order, Limit> : public sub_container<Limit, container_t<Order, std::tuple<std::decay_t<Args>...>, R>> {
private:
	using args_type = std::tuple<std::decay_t<Args>...>;
	using return_type = R;
	container_t<Order, args_type, return_type> values_map;
	std::function<R(Args...)> func;

public:
	memoization() = default;
	template <typename F>
	memoization(F &&f) : func{std::forward<F>(f)} {}
	memoization(const memoization &) = default;
	memoization(memoization &&) noexcept = default;
	memoization &operator=(const memoization &) = default;
	memoization &operator=(memoization &&) noexcept = default;

	R operator()(Args... args) {
		args_type key(std::forward<Args>(args)...);
		auto it = values_map.find(key);
		if (it != values_map.end()) {
			if constexpr (!std::same_as<Limit, unlimited>) {
				auto node = std::find(this->insertion_order.begin(),
									  this->insertion_order.end(),
									  it);
				if (node != this->insertion_order.end()) {
					this->insertion_order.splice(this->insertion_order.end(), this->insertion_order, node);
				}
			}
			return it->second;
		}

		R result = func(std::forward<Args>(args)...);
		if constexpr (!std::same_as<Limit, unlimited>) {
			if (values_map.size() >= this->size && !values_map.empty() && !this->insertion_order.empty()) {
				values_map.erase(this->insertion_order.front());
				this->insertion_order.pop_front();
			}
			auto [new_it, bool_value] = values_map.emplace(std::move(key), result);
			this->insertion_order.push_back(new_it);
		} else {
			values_map.emplace(std::move(key), result);
		}
		return result;
	}
};
template <typename F>
memoization(F) -> memoization<typename function_traits<F>::signature>;

template <typename F>
auto make_memoization(F &&f) {
	using decay_f = std::decay_t<F>;
	using sig = typename function_traits<decay_f>::signature;

	return memoization<sig>(std::function<sig>(std::forward<F>(f)));
}

} // namespace opt_utils

namespace std {

template <typename... Ts>
struct hash<tuple<Ts...>> {
	size_t operator()(tuple<Ts...> const &t) const noexcept {
		size_t seed = 0;
		apply([&](auto const &...elems) {
			((
				 seed ^= std::hash<std::decay_t<decltype(elems)>>{}(elems) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2)),
			 ...);
		},
			  t);

		return seed;
	}
};

} // namespace std
