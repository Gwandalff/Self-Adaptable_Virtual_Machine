package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Not extends EObject, AcceptInterface, Expression {
	Expression getExpression();

	void setExpression(Expression value);
}
