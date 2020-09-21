package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.ArrayTypeRef;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ArrayTypeRefOperation;

public class ArrayTypeRefOperationImpl extends SingleTypeRefOperationImpl implements ArrayTypeRefOperation {
	private final ArrayTypeRef it;

	private final VisitorInterface vis;

	public ArrayTypeRefOperationImpl(ArrayTypeRef it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
