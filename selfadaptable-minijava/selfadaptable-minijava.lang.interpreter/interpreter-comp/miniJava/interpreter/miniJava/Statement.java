package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

import miniJava.interpreter.IAdaptationNode;

public interface Statement extends EObject, IAdaptationNode {
	void evaluateStatement(State state);
}
