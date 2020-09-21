package miniJava.visitor.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface MethodCall extends EObject, AcceptInterface, Expression {
	Expression getReceiver();

	void setReceiver(Expression value);

	Method getMethod();

	void setMethod(Method value);

	EList<Expression> getArgs();
}
