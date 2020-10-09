package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Parameter extends EObject, Symbol {
	boolean compare(Parameter other);
}
