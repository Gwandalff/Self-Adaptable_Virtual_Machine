package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface BooleanValue extends EObject, AcceptInterface, Value {
	boolean isValue();

	void setValue(boolean value);
}
