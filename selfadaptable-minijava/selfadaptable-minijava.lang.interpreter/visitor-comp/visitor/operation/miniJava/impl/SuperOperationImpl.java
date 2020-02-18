package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Super;
import visitor.VisitorInterface;
import visitor.operation.miniJava.SuperOperation;

public class SuperOperationImpl extends ExpressionOperationImpl implements SuperOperation {
	private final Super it;

	private final VisitorInterface vis;

	public SuperOperationImpl(Super it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
