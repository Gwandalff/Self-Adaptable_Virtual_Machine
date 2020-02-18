package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Interface;
import visitor.VisitorInterface;
import visitor.operation.miniJava.InterfaceOperation;

public class InterfaceOperationImpl extends TypeDeclarationOperationImpl implements InterfaceOperation {
	private final Interface it;

	private final VisitorInterface vis;

	public InterfaceOperationImpl(Interface it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
