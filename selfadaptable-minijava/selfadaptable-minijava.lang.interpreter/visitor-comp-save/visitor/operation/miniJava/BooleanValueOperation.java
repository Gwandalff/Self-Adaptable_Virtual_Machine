package visitor.operation.miniJava;

import java.lang.String;
import miniJava.visitor.miniJava.Value;

public interface BooleanValueOperation extends ValueOperation {
	String customToString();

	Value copyj();
}
