package visitor.operation.miniJava;

import java.lang.String;
import miniJava.visitor.miniJava.Value;

public interface IntegerValueOperation extends ValueOperation {
	String customToString();

	Value copyj();
}
