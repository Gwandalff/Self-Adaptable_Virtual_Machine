package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface FieldAccess extends EObject, AcceptInterface, Expression {
	Expression getReceiver();

	void setReceiver(Expression value);

	Field getField();

	void setField(Field value);
}
