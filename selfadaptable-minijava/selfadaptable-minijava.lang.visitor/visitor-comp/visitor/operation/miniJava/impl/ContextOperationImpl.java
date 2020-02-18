package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Symbol;
import miniJava.visitor.miniJava.SymbolBinding;
import minijava.MapService;
import org.eclipse.emf.common.util.EMap;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.LogService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ContextOperation;

public class ContextOperationImpl implements ContextOperation {
	private final Context it;

	private final VisitorInterface vis;

	public ContextOperationImpl(Context it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}

	public SymbolBinding findBinding(Symbol symbol) {
		SymbolBinding result;
		if(!(MapService.containsKey((EMap) (this.it.getCache()), (Symbol) (symbol)))) {
			SymbolBinding binding = ((SymbolBinding) (CollectionService.head(CollectionService.select(this.it.getBindings(), (x) -> EqualService.equals((x.getSymbol()), (symbol))))));
			if(!EqualService.equals((binding), (null))) {
				MapService.put((EMap) (this.it.getCache()), (Symbol) (symbol), (SymbolBinding) (binding));
			}
			else {
				if(!EqualService.equals((this.it.getParentContext()), (null))) {
					SymbolBinding binding2 = ((SymbolBinding) (((ContextOperation)this.it.getParentContext().accept(vis)).findBinding((Symbol) (symbol))));
					MapService.put((EMap) (this.it.getCache()), (Symbol) (symbol), (SymbolBinding) (binding2));
				}
				else {
					LogService.log(("No binding for symbol ") + (symbol));
				}
			}
		}
		result = ((SymbolBinding) (this.it.getCache().get((Symbol) (symbol))));
		return result;
	}

	public Context findCurrentContext() {
		Context result;
		if(!EqualService.equals((this.it.getChildContext()), (null))) {
			result = ((Context) (((ContextOperation)this.it.getChildContext().accept(vis)).findCurrentContext()));
		}
		else {
			result = ((Context) (this.it));
		}
		return result;
	}
}
