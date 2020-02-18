package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ArrayAccess extends EObject, AcceptInterface, Expression {
	Expression getObject();

	void setObject(Expression value);

	Expression getIndex();

	void setIndex(Expression value);
}
