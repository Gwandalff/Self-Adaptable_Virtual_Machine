package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Null extends EObject, Expression {
	Value evaluateExpression(State state);
}
