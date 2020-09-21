package miniJava.visitor.miniJava;

import java.lang.String;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface NamedElement extends EObject, AcceptInterface {
	String getName();

	void setName(String value);
}
