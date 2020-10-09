package miniJava.interpreter.miniJava;

import java.lang.Override;
import java.lang.String;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.eclipse.emf.common.util.Enumerator;

public enum AccessLevel implements Enumerator {
	PRIVATE(0, "PRIVATE", "private"),

	PROTECTED(1, "PROTECTED", "protected"),

	PUBLIC(2, "PUBLIC", "public");

	public static final int PRIVATE_VALUE = 0;

	public static final int PROTECTED_VALUE = 1;

	public static final int PUBLIC_VALUE = 2;

	private static final AccessLevel[] VALUES_ARRAY = new AccessLevel[] { PRIVATE, PROTECTED, PUBLIC };

	public static final List<AccessLevel> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	private final int value;

	private final String name;

	private final String literal;

	private AccessLevel(int value, String name, String literal) {
		this.value = value;
		this.name = name;
		this.literal = literal;
	}

	public static AccessLevel get(int value) {
		switch (value) {
			case PRIVATE_VALUE: return PRIVATE;
			case PROTECTED_VALUE: return PROTECTED;
			case PUBLIC_VALUE: return PUBLIC;
		}
		return null;
	}

	public static AccessLevel get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			AccessLevel result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	public static AccessLevel getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			AccessLevel result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	public int getValue() {
		return value;
	}

	public String getLiteral() {
		return literal;
	}

	public String getName() {
		return name;
	}

	@Override
	public String toString() {
		return literal;
	}
}
