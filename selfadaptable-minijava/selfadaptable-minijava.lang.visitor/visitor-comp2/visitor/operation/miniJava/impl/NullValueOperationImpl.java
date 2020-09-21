package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.NullValue;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.NullValueOperation;

public class NullValueOperationImpl extends ValueOperationImpl implements NullValueOperation {
	private final NullValue it;

	private final VisitorInterface vis;

	public NullValueOperationImpl(NullValue it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value copyj() {
		Value result;
		NullValue tmp = ((NullValue) (MiniJavaFactory.eINSTANCE.createNullValue()));
		result = ((NullValue) (tmp));
		return result;
	}
}
