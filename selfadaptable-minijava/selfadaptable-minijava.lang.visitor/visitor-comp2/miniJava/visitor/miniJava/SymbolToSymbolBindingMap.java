package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface SymbolToSymbolBindingMap extends EObject, AcceptInterface {
	Symbol getKey();

	void setKey(Symbol value);

	SymbolBinding getValue();

	void setValue(SymbolBinding value);
}
