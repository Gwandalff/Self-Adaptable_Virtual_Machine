package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.ObjectInstance;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ObjectInstanceOperation;

public class ObjectInstanceOperationImpl implements ObjectInstanceOperation {
	private final ObjectInstance it;

	private final VisitorInterface vis;

	public ObjectInstanceOperationImpl(ObjectInstance it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
