package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.FieldBinding;
import visitor.VisitorInterface;
import visitor.operation.miniJava.FieldBindingOperation;

public class FieldBindingOperationImpl implements FieldBindingOperation {
	private final FieldBinding it;

	private final VisitorInterface vis;

	public FieldBindingOperationImpl(FieldBinding it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
