package miniJava.interpreter.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;

public interface OutputStream extends EObject {
	EList<String> getStream();
}
