package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface PrintStatement extends EObject, AcceptInterface, Statement {
	Expression getExpression();

	void setExpression(Expression value);
}
