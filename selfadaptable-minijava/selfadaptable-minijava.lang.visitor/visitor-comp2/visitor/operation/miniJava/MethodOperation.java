package visitor.operation.miniJava;

import miniJava.visitor.miniJava.Clazz;
import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.State;

public interface MethodOperation extends MemberOperation {
	void evaluateStatement(State state);

	Method findOverride(Clazz c);

	void call(State state);
}
