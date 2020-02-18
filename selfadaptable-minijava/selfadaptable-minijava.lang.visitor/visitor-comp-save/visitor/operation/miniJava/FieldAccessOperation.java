package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;

public interface FieldAccessOperation extends ExpressionOperation {
	Value evaluateExpression(State state);
}
