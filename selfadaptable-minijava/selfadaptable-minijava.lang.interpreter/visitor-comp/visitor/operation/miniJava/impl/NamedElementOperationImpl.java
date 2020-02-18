package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.NamedElement;
import visitor.VisitorInterface;
import visitor.operation.miniJava.NamedElementOperation;

public class NamedElementOperationImpl implements NamedElementOperation {
	private final NamedElement it;

	private final VisitorInterface vis;

	public NamedElementOperationImpl(NamedElement it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
