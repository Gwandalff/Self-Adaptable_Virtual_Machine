package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface SuperiorOrEqual extends EObject, Expression {
	Expression getLeft();

	void setLeft(Expression value);

	Expression getRight();

	void setRight(Expression value);

	Value evaluateExpression(State state);
}
