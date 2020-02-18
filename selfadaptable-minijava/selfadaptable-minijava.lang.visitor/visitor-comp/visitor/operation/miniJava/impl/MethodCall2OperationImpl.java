package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.MethodCall2;
import visitor.VisitorInterface;
import visitor.operation.miniJava.MethodCall2Operation;

public class MethodCall2OperationImpl extends CallOperationImpl implements MethodCall2Operation {
	private final MethodCall2 it;

	private final VisitorInterface vis;

	public MethodCall2OperationImpl(MethodCall2 it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
