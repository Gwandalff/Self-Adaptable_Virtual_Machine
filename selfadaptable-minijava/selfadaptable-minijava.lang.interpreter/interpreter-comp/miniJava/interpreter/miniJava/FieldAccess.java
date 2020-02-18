package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface FieldAccess extends EObject, Expression {
	Expression getReceiver();

	void setReceiver(Expression value);

	Field getField();

	void setField(Field value);

	Value evaluateExpression(State state);
}
