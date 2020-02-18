package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface SymbolRef extends EObject, AcceptInterface, Expression {
	Symbol getSymbol();

	void setSymbol(Symbol value);
}
