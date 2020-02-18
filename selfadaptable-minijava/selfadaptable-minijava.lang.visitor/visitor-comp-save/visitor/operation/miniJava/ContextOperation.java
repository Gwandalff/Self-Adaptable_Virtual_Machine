package visitor.operation.miniJava;

import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Symbol;
import miniJava.visitor.miniJava.SymbolBinding;

public interface ContextOperation {
	SymbolBinding findBinding(Symbol symbol);

	Context findCurrentContext();
}
