package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.VoidTypeRef;
import visitor.VisitorInterface;
import visitor.operation.miniJava.VoidTypeRefOperation;

public class VoidTypeRefOperationImpl extends SingleTypeRefOperationImpl implements VoidTypeRefOperation {
	private final VoidTypeRef it;

	private final VisitorInterface vis;

	public VoidTypeRefOperationImpl(VoidTypeRef it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
