package miniJava.visitor.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.EMap;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Context extends EObject, AcceptInterface {
	EList<SymbolBinding> getBindings();

	Context getParentContext();

	void setParentContext(Context value);

	Context getChildContext();

	void setChildContext(Context value);

	EMap<Symbol, SymbolBinding> getCache();
}
