package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface NewArray extends EObject, Expression {
	TypeRef getType();

	void setType(TypeRef value);

	Expression getSize();

	void setSize(Expression value);

	Value evaluateExpression(State state);
}
