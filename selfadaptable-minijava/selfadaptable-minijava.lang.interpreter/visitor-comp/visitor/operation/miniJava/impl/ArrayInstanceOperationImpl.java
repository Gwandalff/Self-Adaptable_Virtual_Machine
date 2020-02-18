package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.ArrayInstance;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ArrayInstanceOperation;

public class ArrayInstanceOperationImpl implements ArrayInstanceOperation {
	private final ArrayInstance it;

	private final VisitorInterface vis;

	public ArrayInstanceOperationImpl(ArrayInstance it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
