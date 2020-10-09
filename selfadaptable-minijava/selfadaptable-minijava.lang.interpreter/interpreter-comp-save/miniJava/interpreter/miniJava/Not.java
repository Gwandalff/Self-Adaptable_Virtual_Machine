package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Not extends EObject, Expression {
	Expression getExpression();

	void setExpression(Expression value);

	Value evaluateExpression(State state);
}
