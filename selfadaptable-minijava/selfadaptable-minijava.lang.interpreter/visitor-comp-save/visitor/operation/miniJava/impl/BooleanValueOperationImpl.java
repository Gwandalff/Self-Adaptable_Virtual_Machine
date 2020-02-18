package visitor.operation.miniJava.impl;

import java.lang.String;
import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BooleanValueOperation;

public class BooleanValueOperationImpl extends ValueOperationImpl implements BooleanValueOperation {
	private final BooleanValue it;

	private final VisitorInterface vis;

	public BooleanValueOperationImpl(BooleanValue it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public String customToString() {
		String result;
		result = ((String) ((this.it.isValue()) + ("")));
		return result;
	}

	public Value copyj() {
		Value result;
		BooleanValue tmp = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
		tmp.setValue(this.it.isValue());
		result = ((BooleanValue) (tmp));
		return result;
	}
}
