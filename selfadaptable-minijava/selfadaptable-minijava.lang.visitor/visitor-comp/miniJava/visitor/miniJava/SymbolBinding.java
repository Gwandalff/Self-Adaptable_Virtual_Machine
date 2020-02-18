package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface SymbolBinding extends EObject, AcceptInterface {
	Value getValue();

	void setValue(Value value);

	Symbol getSymbol();

	void setSymbol(Symbol value);
}
