package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Expression extends EObject, Statement, Assignee {
	void evaluateStatement(State state);

	Value evaluateExpression(State state);
}
