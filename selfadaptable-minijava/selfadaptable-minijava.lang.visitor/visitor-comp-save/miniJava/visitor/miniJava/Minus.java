package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Minus extends EObject, AcceptInterface, Expression {
	Expression getLeft();

	void setLeft(Expression value);

	Expression getRight();

	void setRight(Expression value);
}
