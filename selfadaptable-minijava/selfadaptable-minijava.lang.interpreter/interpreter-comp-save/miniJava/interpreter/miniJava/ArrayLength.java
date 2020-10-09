package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface ArrayLength extends EObject, Expression {
	Expression getArray();

	void setArray(Expression value);

	Value evaluateExpression(State state);
}
