package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface NewCall extends EObject, Call {
	NewObject getNewz();

	void setNewz(NewObject value);
}
