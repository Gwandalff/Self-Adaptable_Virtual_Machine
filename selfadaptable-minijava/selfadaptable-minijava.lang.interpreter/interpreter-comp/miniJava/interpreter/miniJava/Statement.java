package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

import miniJava.interpreter.IDynamicSubject;

public interface Statement extends EObject, IDynamicSubject {
	void evaluateStatement(State state);
}
