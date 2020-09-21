package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface NewCall extends EObject, AcceptInterface, Call {
	NewObject getNewz();

	void setNewz(NewObject value);
}
