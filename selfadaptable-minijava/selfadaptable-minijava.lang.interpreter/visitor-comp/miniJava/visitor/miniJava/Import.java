package miniJava.visitor.miniJava;

import java.lang.String;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Import extends EObject, AcceptInterface {
	String getImportedNamespace();

	void setImportedNamespace(String value);
}
