package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Return extends EObject, AcceptInterface, Statement {
	Expression getExpression();

	void setExpression(Expression value);
}
