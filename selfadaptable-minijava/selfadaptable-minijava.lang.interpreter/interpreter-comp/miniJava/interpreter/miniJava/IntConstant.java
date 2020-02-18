package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface IntConstant extends EObject, Expression {
	int getValue();

	void setValue(int value);

	Value evaluateExpression(State state);
}
