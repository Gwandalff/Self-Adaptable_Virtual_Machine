package visitor.operation.miniJava;

import java.lang.String;
import miniJava.visitor.miniJava.Value;

public interface ArrayRefValueOperation extends ValueOperation {
	Value copyj();

	String customToString();
}
