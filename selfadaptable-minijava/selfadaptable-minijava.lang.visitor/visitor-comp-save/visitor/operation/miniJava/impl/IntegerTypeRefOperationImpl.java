package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.IntegerTypeRef;
import visitor.VisitorInterface;
import visitor.operation.miniJava.IntegerTypeRefOperation;

public class IntegerTypeRefOperationImpl extends SingleTypeRefOperationImpl implements IntegerTypeRefOperation {
	private final IntegerTypeRef it;

	private final VisitorInterface vis;

	public IntegerTypeRefOperationImpl(IntegerTypeRef it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
