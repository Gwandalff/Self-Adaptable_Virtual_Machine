package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface MethodCall2 extends EObject, AcceptInterface, Call {
	MethodCall getMethodcall();

	void setMethodcall(MethodCall value);
}
