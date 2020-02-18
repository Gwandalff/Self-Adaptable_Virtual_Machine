package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.SymbolToSymbolBindingMap;
import visitor.VisitorInterface;
import visitor.operation.miniJava.SymbolToSymbolBindingMapOperation;

public class SymbolToSymbolBindingMapOperationImpl implements SymbolToSymbolBindingMapOperation {
	private final SymbolToSymbolBindingMap it;

	private final VisitorInterface vis;

	public SymbolToSymbolBindingMapOperationImpl(SymbolToSymbolBindingMap it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
