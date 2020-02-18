package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.SymbolBinding;
import visitor.VisitorInterface;
import visitor.operation.miniJava.SymbolBindingOperation;

public class SymbolBindingOperationImpl implements SymbolBindingOperation {
	private final SymbolBinding it;

	private final VisitorInterface vis;

	public SymbolBindingOperationImpl(SymbolBinding it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
