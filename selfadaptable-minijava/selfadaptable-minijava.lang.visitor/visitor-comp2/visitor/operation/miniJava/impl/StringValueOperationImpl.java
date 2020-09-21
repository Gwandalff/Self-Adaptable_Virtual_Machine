package visitor.operation.miniJava.impl;

import java.lang.String;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.StringValue;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.StringValueOperation;

public class StringValueOperationImpl extends ValueOperationImpl implements StringValueOperation {
	private final StringValue it;

	private final VisitorInterface vis;

	public StringValueOperationImpl(StringValue it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public String customToString() {
		String result;
		result = ((String) (this.it.getValue().toString()));
		return result;
	}

	public Value copyj() {
		Value result;
		StringValue tmp = ((StringValue) (MiniJavaFactory.eINSTANCE.createStringValue()));
		tmp.setValue(this.it.getValue());
		result = ((StringValue) (tmp));
		return result;
	}
}
