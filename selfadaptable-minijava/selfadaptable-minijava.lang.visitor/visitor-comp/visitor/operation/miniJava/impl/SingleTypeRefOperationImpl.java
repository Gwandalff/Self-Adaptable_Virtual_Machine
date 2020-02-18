package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.SingleTypeRef;
import visitor.VisitorInterface;
import visitor.operation.miniJava.SingleTypeRefOperation;

public class SingleTypeRefOperationImpl extends TypeRefOperationImpl implements SingleTypeRefOperation {
	private final SingleTypeRef it;

	private final VisitorInterface vis;

	public SingleTypeRefOperationImpl(SingleTypeRef it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
