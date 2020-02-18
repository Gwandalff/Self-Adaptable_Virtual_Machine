package miniJava.interpreter.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;

public interface MethodCall extends EObject, Expression {
	Expression getReceiver();

	void setReceiver(Expression value);

	Method getMethod();

	void setMethod(Method value);

	EList<Expression> getArgs();

	Value evaluateExpression(State state);

	void call(Method realMethod, State state);
}
