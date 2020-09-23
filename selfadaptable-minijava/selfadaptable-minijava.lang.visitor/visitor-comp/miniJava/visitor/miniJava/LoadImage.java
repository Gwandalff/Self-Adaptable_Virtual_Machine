package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface LoadImage extends EObject, AcceptInterface, Expression {
	Expression getPath();

	void setPath(Expression value);
}
