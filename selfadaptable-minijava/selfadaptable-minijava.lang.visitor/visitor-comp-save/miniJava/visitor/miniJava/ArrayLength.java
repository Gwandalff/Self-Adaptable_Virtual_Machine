package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ArrayLength extends EObject, AcceptInterface, Expression {
	Expression getArray();

	void setArray(Expression value);
}
