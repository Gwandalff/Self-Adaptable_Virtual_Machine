package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface SymbolBinding extends EObject {
	Value getValue();

	void setValue(Value value);

	Symbol getSymbol();

	void setSymbol(Symbol value);
}
