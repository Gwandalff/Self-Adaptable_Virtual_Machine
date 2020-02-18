package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface SymbolRef extends EObject, Expression {
	Symbol getSymbol();

	void setSymbol(Symbol value);

	Value evaluateExpression(State state);
}
