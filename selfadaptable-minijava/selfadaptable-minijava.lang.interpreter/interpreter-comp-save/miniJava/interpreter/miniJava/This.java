package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface This extends EObject, Expression {
	Value evaluateExpression(State state);
}
