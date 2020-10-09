package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface LoadImage extends EObject, Expression {
	Expression getPath();

	void setPath(Expression value);

	Value evaluateExpression(State state);
}
