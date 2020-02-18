package visitor.operation.miniJava.impl;

import java.lang.String;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.IntegerValueOperation;

public class IntegerValueOperationImpl extends ValueOperationImpl implements IntegerValueOperation {
	private final IntegerValue it;

	private final VisitorInterface vis;

	public IntegerValueOperationImpl(IntegerValue it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public String customToString() {
		String result;
		result = ((String) ((this.it.getValue()) + ("")));
		return result;
	}

	public Value copyj() {
		Value result;
		IntegerValue tmp = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
		tmp.setValue(this.it.getValue());
		result = ((IntegerValue) (tmp));
		return result;
	}
}
