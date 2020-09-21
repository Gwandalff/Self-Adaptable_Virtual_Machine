package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface IntConstant extends EObject, AcceptInterface, Expression {
	int getValue();

	void setValue(int value);
}
