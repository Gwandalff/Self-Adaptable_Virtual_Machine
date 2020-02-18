package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Field;
import visitor.VisitorInterface;
import visitor.operation.miniJava.FieldOperation;

public class FieldOperationImpl extends MemberOperationImpl implements FieldOperation {
	private final Field it;

	private final VisitorInterface vis;

	public FieldOperationImpl(Field it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
