package miniJava.visitor.miniJava;

import java.lang.String;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface StringValue extends EObject, AcceptInterface, Value {
	String getValue();

	void setValue(String value);
}
