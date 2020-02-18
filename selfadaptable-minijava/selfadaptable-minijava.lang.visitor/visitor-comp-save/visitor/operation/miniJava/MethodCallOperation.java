package visitor.operation.miniJava;

import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;

public interface MethodCallOperation extends ExpressionOperation {
	Value evaluateExpression(State state);

	void call(Method realMethod, State state);
}
