package visitor.operation.miniJava.impl;

import java.lang.String;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.ObjectRefValue;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ObjectRefValueOperation;

public class ObjectRefValueOperationImpl extends ValueOperationImpl implements ObjectRefValueOperation {
	private final ObjectRefValue it;

	private final VisitorInterface vis;

	public ObjectRefValueOperationImpl(ObjectRefValue it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public String customToString() {
		String result;
		result = ((String) (this.it.getInstance().toString()));
		return result;
	}

	public Value copyj() {
		Value result;
		ObjectRefValue tmp = ((ObjectRefValue) (MiniJavaFactory.eINSTANCE.createObjectRefValue()));
		tmp.setInstance(this.it.getInstance());
		result = ((ObjectRefValue) (tmp));
		return result;
	}
}
