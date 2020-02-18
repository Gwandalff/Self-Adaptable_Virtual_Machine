package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface NewArray extends EObject, AcceptInterface, Expression {
	TypeRef getType();

	void setType(TypeRef value);

	Expression getSize();

	void setSize(Expression value);
}
