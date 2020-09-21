package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.ArrayLength;
import miniJava.visitor.miniJava.ArrayRefValue;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ArrayLengthOperation;
import visitor.operation.miniJava.ExpressionOperation;

public class ArrayLengthOperationImpl extends ExpressionOperationImpl implements ArrayLengthOperation {
	private final ArrayLength it;

	private final VisitorInterface vis;

	public ArrayLengthOperationImpl(ArrayLength it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		ArrayRefValue arrayRef = ((ArrayRefValue) (((ExpressionOperation)this.it.getArray().accept(vis)).evaluateExpression((State) (state))));
		int size = ((int) (arrayRef.getInstance().getSize()));
		IntegerValue tmp = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
		tmp.setValue(size);
		result = ((IntegerValue) (tmp));
		return result;
	}
}
