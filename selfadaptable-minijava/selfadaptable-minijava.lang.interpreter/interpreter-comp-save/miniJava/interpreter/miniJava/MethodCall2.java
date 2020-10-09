package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface MethodCall2 extends EObject, Call {
	MethodCall getMethodcall();

	void setMethodcall(MethodCall value);
}
