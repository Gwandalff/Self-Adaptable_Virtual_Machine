package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Field extends EObject, AcceptInterface, Member {
	Expression getDefaultValue();

	void setDefaultValue(Expression value);
}
