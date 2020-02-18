package visitor.operation.miniJava;

import miniJava.visitor.miniJava.Parameter;

public interface ParameterOperation extends SymbolOperation {
	boolean compare(Parameter other);
}
