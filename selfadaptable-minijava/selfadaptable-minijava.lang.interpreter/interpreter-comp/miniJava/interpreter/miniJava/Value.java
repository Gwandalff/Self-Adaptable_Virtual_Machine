package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Value extends EObject {
	String customToString();

	Value copyj();
}
