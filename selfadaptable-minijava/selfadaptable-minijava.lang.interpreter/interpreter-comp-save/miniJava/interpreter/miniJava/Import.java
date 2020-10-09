package miniJava.interpreter.miniJava;

import java.lang.String;
import org.eclipse.emf.ecore.EObject;

public interface Import extends EObject {
	String getImportedNamespace();

	void setImportedNamespace(String value);
}
