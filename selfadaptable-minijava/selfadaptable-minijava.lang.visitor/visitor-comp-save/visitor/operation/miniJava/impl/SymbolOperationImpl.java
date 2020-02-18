package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Symbol;
import visitor.VisitorInterface;
import visitor.operation.miniJava.SymbolOperation;

public class SymbolOperationImpl extends TypedDeclarationOperationImpl implements SymbolOperation {
	private final Symbol it;

	private final VisitorInterface vis;

	public SymbolOperationImpl(Symbol it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
