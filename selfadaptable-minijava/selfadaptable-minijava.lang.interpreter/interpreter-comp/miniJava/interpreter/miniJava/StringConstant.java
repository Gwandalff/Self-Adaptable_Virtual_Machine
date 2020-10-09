package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface StringConstant extends EObject, Expression {
	String getValue();

	void setValue(String value);

	Value evaluateExpression(State state);
}
