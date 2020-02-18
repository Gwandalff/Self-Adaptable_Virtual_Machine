package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.VariableDeclaration;
import visitor.VisitorInterface;
import visitor.operation.miniJava.VariableDeclarationOperation;

public class VariableDeclarationOperationImpl extends SymbolOperationImpl implements VariableDeclarationOperation {
	private final VariableDeclaration it;

	private final VisitorInterface vis;

	public VariableDeclarationOperationImpl(VariableDeclaration it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
