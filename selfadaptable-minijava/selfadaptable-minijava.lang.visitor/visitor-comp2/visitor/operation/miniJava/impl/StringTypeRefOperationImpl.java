package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.StringTypeRef;
import visitor.VisitorInterface;
import visitor.operation.miniJava.StringTypeRefOperation;

public class StringTypeRefOperationImpl extends SingleTypeRefOperationImpl implements StringTypeRefOperation {
	private final StringTypeRef it;

	private final VisitorInterface vis;

	public StringTypeRefOperationImpl(StringTypeRef it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
