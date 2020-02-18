package minijava;

import java.util.Map;
import java.util.function.Function;

import org.eclipse.emf.common.util.BasicEMap;
import org.eclipse.emf.common.util.EMap;

public class MapService {



	public static <K, V> void put(EMap<K, V> map, K key, V value) {
		map.put(key, value);
	}

	public static <K, V> void putAll(EMap<K, V> map, EMap<? extends K, ? extends V> all) {
		map.putAll(all);
	}

	public static <K, V> void replaceWith(EMap<K, V> map, EMap<? extends K, ? extends V> all) {
		map.clear();
		map.putAll(all);
	}

	public static <K, V> V getFromMap(EMap<K, V> map, K key) {
		return map.get(key);
	}

	public static <K, V> String mapToString(EMap<K, V> map, Function<Map.Entry<K, V>, String> f) {
		String ret = "";
		for (Map.Entry<K, V> x : map.entrySet()) {
			ret += f.apply(x);
		}
		return ret;
	}

//	public static <K, KV> boolean containsKey(EList<KV> map, K key) {
//		return map.stream().filter(x -> ((Map.Entry<K, ?>) x).getKey().equals(key)).findAny().isPresent();
//	}

	public static <K, V> boolean containsKey(EMap<K, V> map, K key) {
		return map.containsKey(key);
	}

	public static <K, V> EMap<K, V> newMap(Object self) {
		return new BasicEMap<>();
	}

//	public static <K, V, KV> V get(EList<KV> map, K key) {
//		return (V) map.stream().filter(x -> ((Map.Entry<K, V>) x).getKey().equals(key)).findAny().get();
//	}
}
