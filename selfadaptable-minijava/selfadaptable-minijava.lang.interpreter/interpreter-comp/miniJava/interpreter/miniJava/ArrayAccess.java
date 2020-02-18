package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface ArrayAccess extends EObject, Expression {
	Expression getObject();

	void setObject(Expression value);

	Expression getIndex();

	void setIndex(Expression value);

	Value evaluateExpression(State state);
}
