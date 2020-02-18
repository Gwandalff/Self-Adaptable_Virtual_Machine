package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.BooleanTypeRef;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BooleanTypeRefOperation;

public class BooleanTypeRefOperationImpl extends SingleTypeRefOperationImpl implements BooleanTypeRefOperation {
	private final BooleanTypeRef it;

	private final VisitorInterface vis;

	public BooleanTypeRefOperationImpl(BooleanTypeRef it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
